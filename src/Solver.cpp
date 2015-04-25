#include "Solver.h"

#include "base/Parallel.h"
#include "base/SIMD.h"

#include "Configuration.h"

const float kProductiveImpulse = 1e-4f;
const float kFrictionCoefficient = 0.3f;

const int kIslandMinSize = 256;

Solver::Solver()
{
}

NOINLINE void Solver::SolveJoints(WorkQueue& queue, RigidBody* bodies, int bodiesCount, ContactPoint* contactPoints, const Configuration& configuration)
{
    switch (configuration.solveMode)
    {
    case Configuration::Solve_AVX2:
    #ifdef __AVX2__
        SolveJoints_AVX2(queue, bodies, bodiesCount, contactPoints, configuration);
        break;
    #endif

    case Configuration::Solve_SSE2:
    #ifdef __SSE2__
        SolveJoints_SSE2(queue, bodies, bodiesCount, contactPoints, configuration);
        break;
    #endif

    case Configuration::Solve_Scalar:
        SolveJoints_Scalar(queue, bodies, bodiesCount, contactPoints, configuration);
        break;

    default:
        assert(!"Unknown solver mode");
        break;
    }
}

NOINLINE void Solver::SolveJoints_Scalar(WorkQueue& queue, RigidBody* bodies, int bodiesCount, ContactPoint* contactPoints, const Configuration& configuration)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJoints_Scalar", -1);

    SolveJoints(queue, joint_packed1, bodies, bodiesCount, contactPoints, configuration);
}

#ifdef __SSE2__
NOINLINE void Solver::SolveJoints_SSE2(WorkQueue& queue, RigidBody* bodies, int bodiesCount, ContactPoint* contactPoints, const Configuration& configuration)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJoints_SSE2", -1);

    SolveJoints(queue, joint_packed4, bodies, bodiesCount, contactPoints, configuration);
}
#endif

#ifdef __AVX2__
NOINLINE void Solver::SolveJoints_AVX2(WorkQueue& queue, RigidBody* bodies, int bodiesCount, ContactPoint* contactPoints, const Configuration& configuration)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJoints_AVX2", -1);

    SolveJoints(queue, joint_packed8, bodies, bodiesCount, contactPoints, configuration);
}
#endif

template <int N>
void Solver::SolveJoints(WorkQueue& queue, AlignedArray<ContactJointPacked<N>>& joint_packed, RigidBody* bodies, int bodiesCount, ContactPoint* contactPoints, const Configuration& configuration)
{
    PrepareBodies(bodies, bodiesCount);

    if (configuration.islandMode != Configuration::Island_Single)
    {
        int jointCountAligned = GatherIslands(bodies, bodiesCount, N);

        joint_packed.resize(jointCountAligned);
        jointGroup_joints.resize(jointCountAligned);
        jointGroup_bodies.resize(bodiesCount);

        for (int i = 0; i < bodiesCount; ++i)
            jointGroup_bodies[i] = 0;

        parallelFor(queue, 0, islandCount, 1, [&](int islandIndex, int) {
            int jointsBegin = island_offset[islandIndex];
            int jointsEnd = jointsBegin + island_size[islandIndex];

            SolveJointIsland(queue, joint_packed, jointsBegin, jointsEnd, contactPoints, configuration);
        });
    }
    else
    {
        int jointCount = contactJoints.size;

        joint_index.resize(jointCount);
        joint_packed.resize(jointCount);
        jointGroup_joints.resize(jointCount);
        jointGroup_bodies.resize(bodiesCount);

        for (int i = 0; i < jointCount; ++i)
            joint_index[i] = i;

        for (int i = 0; i < bodiesCount; ++i)
            jointGroup_bodies[i] = 0;

        islandCount = 1;
        islandMaxSize = jointCount;

        SolveJointIsland(queue, joint_packed, 0, jointCount, contactPoints, configuration);
    }

    FinishBodies(bodies, bodiesCount);
}

static bool any(const std::vector<char>& v)
{
    for (auto& e: v)
        if (e)
            return true;

    return false;
}

template <int N>
NOINLINE void Solver::SolveJointIsland(WorkQueue& queue, AlignedArray<ContactJointPacked<N>>& joint_packed, int jointBegin, int jointEnd, ContactPoint* contactPoints, const Configuration& configuration)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointIsland", -1);

    int groupOffset = PrepareJoints(joint_packed, jointBegin, jointEnd, N);

    int batchSize = (configuration.islandMode == Configuration::Island_Sloppy) ? 512 : jointEnd - jointBegin;
    int batchCount = ((groupOffset - jointBegin) + batchSize - 1) / batchSize;

    {
        MICROPROFILE_SCOPEI("Physics", "Prepare", -1);

        parallelFor(queue, 0, batchCount, 1, [&](int batchIndex, int worker) {
            int batchBegin = jointBegin + batchIndex * batchSize;
            int batchEnd = std::min(batchBegin + batchSize, groupOffset);

            RefreshJoints<N>(joint_packed.data, batchBegin, batchEnd, contactPoints);
        });

        RefreshJoints<1>(joint_packed.data, groupOffset, jointEnd, contactPoints);

        parallelFor(queue, 0, batchCount, 1, [&](int batchIndex, int worker) {
            int batchBegin = jointBegin + batchIndex * batchSize;
            int batchEnd = std::min(batchBegin + batchSize, groupOffset);

            PreStepJoints<N>(joint_packed.data, batchBegin, batchEnd);
        });

        PreStepJoints<1>(joint_packed.data, groupOffset, jointEnd);
    }

    {
        MICROPROFILE_SCOPEI("Physics", "Impulse", -1);

        for (int iterationIndex = 0; iterationIndex < configuration.contactIterationsCount; iterationIndex++)
        {
            std::vector<char> productivew(queue.getWorkerCount() + 1);

            parallelFor(queue, 0, batchCount, 1, [&](int batchIndex, int worker) {
                int batchBegin = jointBegin + batchIndex * batchSize;
                int batchEnd = std::min(batchBegin + batchSize, groupOffset);

                productivew[worker] |= SolveJointsImpulses<N>(joint_packed.data, batchBegin, batchEnd, iterationIndex);
            });

            bool productive = any(productivew);

            productive |= SolveJointsImpulses<1>(joint_packed.data, groupOffset, jointEnd, iterationIndex);

            if (!productive) break;
        }
    }

    {
        MICROPROFILE_SCOPEI("Physics", "Displacement", -1);

        for (int iterationIndex = 0; iterationIndex < configuration.penetrationIterationsCount; iterationIndex++)
        {
            std::vector<char> productivew(queue.getWorkerCount() + 1);

            parallelFor(queue, 0, batchCount, 1, [&](int batchIndex, int worker) {
                int batchBegin = jointBegin + batchIndex * batchSize;
                int batchEnd = std::min(batchBegin + batchSize, groupOffset);

                productivew[worker] |= SolveJointsDisplacement<N>(joint_packed.data, batchBegin, batchEnd, iterationIndex);
            });

            bool productive = any(productivew);

            productive |= SolveJointsDisplacement<1>(joint_packed.data, groupOffset, jointEnd, iterationIndex);

            if (!productive) break;
        }
    }

    FinishJoints(joint_packed, jointBegin, jointEnd);
}

NOINLINE int Solver::PrepareIndices(int jointBegin, int jointEnd, int groupSizeTarget)
{
    MICROPROFILE_SCOPEI("Physics", "PrepareIndices", -1);

    if (groupSizeTarget == 1)
        return jointEnd;

    for (int i = jointBegin; i < jointEnd; ++i)
        jointGroup_joints[i] = joint_index[i];

    int tag = 0;

    int remainingJoints = jointEnd - jointBegin;
    int groupOffset = jointBegin;

    while (remainingJoints >= groupSizeTarget)
    {
        // gather a group of N joints with non-overlapping bodies
        int groupSize = 0;

        tag++;

        for (int i = 0; i < remainingJoints && groupSize < groupSizeTarget;)
        {
            int jointIndex = jointGroup_joints[jointBegin + i];
            ContactJoint& joint = contactJoints[jointIndex];

            // TODO: race between static bodies from different islands
            if (jointGroup_bodies[joint.body1Index] < tag && jointGroup_bodies[joint.body2Index] < tag)
            {
                jointGroup_bodies[joint.body1Index] = tag;
                jointGroup_bodies[joint.body2Index] = tag;

                joint_index[groupOffset + groupSize] = jointIndex;
                groupSize++;

                jointGroup_joints[jointBegin + i] = jointGroup_joints[jointBegin + remainingJoints - 1];
                remainingJoints--;
            }
            else
            {
                i++;
            }
        }

        groupOffset += groupSize;

        if (groupSize < groupSizeTarget)
            break;
    }

    // fill in the rest of the joints sequentially - they don't form a group so we'll have to solve them 1 by 1
    for (int i = 0; i < remainingJoints; ++i)
        joint_index[groupOffset + i] = jointGroup_joints[jointBegin + i];

    return groupOffset & ~(groupSizeTarget - 1);
}

static int remap(AlignedArray<int>& table, int index)
{
    int result = index;

    while (result != table[result])
        result = table[result];

    return table[index] = result;
}

NOINLINE int Solver::GatherIslands(RigidBody* bodies, int bodiesCount, int groupSizeTarget)
{
    MICROPROFILE_SCOPEI("Physics", "GatherIslands", -1);

    int jointCount = contactJoints.size;
    int jointCountAligned = jointCount;

    island_remap.resize(bodiesCount);
    island_index.resize(bodiesCount);
    island_indexremap.resize(bodiesCount);
    island_offset.resize(bodiesCount);
    island_offsettemp.resize(bodiesCount);
    island_size.resize(bodiesCount);

    {
        MICROPROFILE_SCOPEI("Physics", "Prepare", -1);

        for (int i = 0; i < bodiesCount; ++i)
        {
            island_remap[i] = (bodies[i].invMass == 0 && bodies[i].invInertia == 0) ? -1 : i;
        }
    }

    {
        MICROPROFILE_SCOPEI("Physics", "Merge", -1);

        for (ContactJoint& j: contactJoints)
        {
            int island1 = island_remap[j.body1Index];
            int island2 = island_remap[j.body2Index];

            if ((island1 | island2) < 0)
                continue;

            int remap1 = remap(island_remap, island1);
            int remap2 = remap(island_remap, island2);

            island_remap[remap1] = remap2;
        }
    }

    {
        MICROPROFILE_SCOPEI("Physics", "Gather", -1);

        islandCount = 0;

        for (int i = 0; i < bodiesCount; ++i)
        {
            island_index[i] = -1;
        }

        for (int i = 0; i < bodiesCount; ++i)
        {
            if (island_remap[i] < 0)
                continue;

            island_remap[i] = remap(island_remap, i);
        }

        for (int i = 0; i < bodiesCount; ++i)
        {
            if (island_remap[i] < 0)
                continue;

            int island = island_remap[i];

            if (island_index[island] < 0)
            {
                island_index[island] = islandCount;
                islandCount++;
            }
        }
    }

    {
        MICROPROFILE_SCOPEI("Physics", "Count", -1);

        for (int i = 0; i < islandCount; ++i)
        {
            island_offset[i] = 0;
        }

        for (ContactJoint& j: contactJoints)
        {
            int island1 = island_remap[j.body1Index];
            int island2 = island_remap[j.body2Index];

            if ((island1 & island2) < 0)
                continue;

            assert(island1 == island2 || ((island1 | island2) < 0 && (island1 & island2) >= 0));
            int island = island1 < 0 ? island2 : island1;

            island_offset[island_index[island]]++;
        }
    }

    {
        MICROPROFILE_SCOPEI("Physics", "Coalesce", -1);

        for (int i = 0; i < islandCount; ++i)
        {
            island_indexremap[i] = i;
        }

        int runningIndex = 0;
        int runningCount = 0;
        int totalCount = 0;

        for (int i = 0; i < islandCount; ++i)
        {
            runningCount += island_offset[i];

            island_indexremap[i] = runningIndex;

            if (runningCount >= kIslandMinSize || i == islandCount - 1)
            {
                int runningCountAligned = (runningCount + groupSizeTarget - 1) & ~(groupSizeTarget - 1);

                island_size[runningIndex] = runningCount;
                island_offset[runningIndex] = totalCount;

                totalCount += runningCountAligned;
                runningCount = 0;
                runningIndex++;
            }
        }

        jointCountAligned = totalCount;
        islandCount = runningIndex;
    }

    joint_index.resize(jointCountAligned);

    {
        MICROPROFILE_SCOPEI("Physics", "Index", -1);

        for (int i = 0; i < islandCount; ++i)
        {
            island_offsettemp[i] = island_offset[i];
        }

        for (int jointIndex = 0; jointIndex < jointCount; ++jointIndex)
        {
            ContactJoint& j = contactJoints[jointIndex];

            int island1 = island_remap[j.body1Index];
            int island2 = island_remap[j.body2Index];

            if ((island1 & island2) < 0)
                continue;

            assert(island1 == island2 || ((island1 | island2) < 0 && (island1 & island2) >= 0));
            int island = island1 < 0 ? island2 : island1;

            joint_index[island_offsettemp[island_indexremap[island_index[island]]]++] = jointIndex;
        }

        islandMaxSize = 0;

        for (int i = 0; i < islandCount; ++i)
        {
            assert(island_offsettemp[i] == island_offset[i] + island_size[i]);

            islandMaxSize = std::max(islandMaxSize, island_size[i]);
        }
    }

    return jointCountAligned;
}

NOINLINE void Solver::PrepareBodies(RigidBody* bodies, int bodiesCount)
{
    MICROPROFILE_SCOPEI("Physics", "PrepareBodies", -1);

    solveBodiesParams.resize(bodiesCount);
    solveBodiesImpulse.resize(bodiesCount);
    solveBodiesDisplacement.resize(bodiesCount);

    for (int i = 0; i < bodiesCount; ++i)
    {
        solveBodiesParams[i].invMass = bodies[i].invMass;
        solveBodiesParams[i].invInertia = bodies[i].invInertia;
        solveBodiesParams[i].coords_pos = bodies[i].coords.pos;
        solveBodiesParams[i].coords_xVector = bodies[i].coords.xVector;
        solveBodiesParams[i].coords_yVector = bodies[i].coords.yVector;

        solveBodiesImpulse[i].velocity = bodies[i].velocity;
        solveBodiesImpulse[i].angularVelocity = bodies[i].angularVelocity;
        solveBodiesImpulse[i].lastIteration = -1;

        solveBodiesDisplacement[i].velocity = bodies[i].displacingVelocity;
        solveBodiesDisplacement[i].angularVelocity = bodies[i].displacingAngularVelocity;
        solveBodiesDisplacement[i].lastIteration = -1;
    }
}

NOINLINE void Solver::FinishBodies(RigidBody* bodies, int bodiesCount)
{
    MICROPROFILE_SCOPEI("Physics", "FinishBodies", -1);

    for (int i = 0; i < bodiesCount; ++i)
    {
        bodies[i].velocity = solveBodiesImpulse[i].velocity;
        bodies[i].angularVelocity = solveBodiesImpulse[i].angularVelocity;

        bodies[i].displacingVelocity = solveBodiesDisplacement[i].velocity;
        bodies[i].displacingAngularVelocity = solveBodiesDisplacement[i].angularVelocity;
    }
}

template <int N>
NOINLINE int Solver::PrepareJoints(AlignedArray<ContactJointPacked<N>>& joint_packed, int jointBegin, int jointEnd, int groupSizeTarget)
{
    MICROPROFILE_SCOPEI("Physics", "PrepareJoints", -1);

    assert(jointBegin % groupSizeTarget == 0);
    assert(jointBegin % N == 0);

    int groupOffset = PrepareIndices(jointBegin, jointEnd, groupSizeTarget);

    {
        MICROPROFILE_SCOPEI("Physics", "CopyJoints", -1);

        for (int i = jointBegin; i < jointEnd; ++i)
        {
            ContactJoint& joint = contactJoints[joint_index[i]];

            ContactJointPacked<N>& jointP = joint_packed[unsigned(i) / N];
            int iP = i & (N - 1);

            jointP.body1Index[iP] = joint.body1Index;
            jointP.body2Index[iP] = joint.body2Index;
            jointP.contactPointIndex[iP] = joint.contactPointIndex;

            jointP.normalLimiter_accumulatedImpulse[iP] = joint.normalLimiter_accumulatedImpulse;
            jointP.frictionLimiter_accumulatedImpulse[iP] = joint.frictionLimiter_accumulatedImpulse;
        }
    }

    return groupOffset;
}

template <int N>
NOINLINE void Solver::FinishJoints(AlignedArray<ContactJointPacked<N>>& joint_packed, int jointBegin, int jointEnd)
{
    MICROPROFILE_SCOPEI("Physics", "FinishJoints", -1);

    assert(jointBegin % N == 0);

    for (int i = jointBegin; i < jointEnd; ++i)
    {
        ContactJoint& joint = contactJoints[joint_index[i]];

        ContactJointPacked<N>& jointP = joint_packed[unsigned(i) / N];
        int iP = i & (N - 1);

        joint.normalLimiter_accumulatedImpulse = jointP.normalLimiter_accumulatedImpulse[iP];
        joint.frictionLimiter_accumulatedImpulse = jointP.frictionLimiter_accumulatedImpulse[iP];
    }
}

template <typename Vf, int N>
static void RefreshLimiter(
    ContactLimiterPacked<N>& limiter, int iP,
    const Vf& n1X, const Vf& n1Y, const Vf& n2X, const Vf& n2Y, const Vf& w1X, const Vf& w1Y, const Vf& w2X, const Vf& w2Y,
    const Vf& body1_invMass, const Vf& body1_invInertia, const Vf& body2_invMass, const Vf& body2_invInertia)
{
    Vf normalProjector1X = n1X;
    Vf normalProjector1Y = n1Y;
    Vf normalProjector2X = n2X;
    Vf normalProjector2Y = n2Y;
    Vf angularProjector1 = n1X * w1Y - n1Y * w1X;
    Vf angularProjector2 = n2X * w2Y - n2Y * w2X;

    Vf compMass1_linearX = normalProjector1X * body1_invMass;
    Vf compMass1_linearY = normalProjector1Y * body1_invMass;
    Vf compMass1_angular = angularProjector1 * body1_invInertia;
    Vf compMass2_linearX = normalProjector2X * body2_invMass;
    Vf compMass2_linearY = normalProjector2Y * body2_invMass;
    Vf compMass2_angular = angularProjector2 * body2_invInertia;

    Vf compMass1 = normalProjector1X * compMass1_linearX + normalProjector1Y * compMass1_linearY + angularProjector1 * compMass1_angular;
    Vf compMass2 = normalProjector2X * compMass2_linearX + normalProjector2Y * compMass2_linearY + angularProjector2 * compMass2_angular;

    Vf compMass = compMass1 + compMass2;

    Vf compInvMass = select(Vf::zero(), Vf::one(1) / compMass, abs(compMass) > Vf::zero());

    store(normalProjector1X, &limiter.normalProjector1X[iP]);
    store(normalProjector1Y, &limiter.normalProjector1Y[iP]);
    store(normalProjector2X, &limiter.normalProjector2X[iP]);
    store(normalProjector2Y, &limiter.normalProjector2Y[iP]);
    store(angularProjector1, &limiter.angularProjector1[iP]);
    store(angularProjector2, &limiter.angularProjector2[iP]);

    store(compMass1_linearX, &limiter.compMass1_linearX[iP]);
    store(compMass1_linearY, &limiter.compMass1_linearY[iP]);
    store(compMass2_linearX, &limiter.compMass2_linearX[iP]);
    store(compMass2_linearY, &limiter.compMass2_linearY[iP]);
    store(compMass1_angular, &limiter.compMass1_angular[iP]);
    store(compMass2_angular, &limiter.compMass2_angular[iP]);
    store(compInvMass, &limiter.compInvMass[iP]);
}

template <int VN, int N>
NOINLINE void Solver::RefreshJoints(ContactJointPacked<N>* joint_packed, int jointBegin, int jointEnd, ContactPoint* contactPoints)
{
    MICROPROFILE_SCOPEI("Physics", "RefreshJoints", -1);

    typedef simd::VNf<VN> Vf;
    typedef simd::VNi<VN> Vi;
    typedef simd::VNb<VN> Vb;

    assert(jointBegin % VN == 0 && jointEnd % VN == 0);

    for (int jointIndex = jointBegin; jointIndex < jointEnd; jointIndex += VN)
    {
        int i = jointIndex;

        ContactJointPacked<N>& jointP = joint_packed[unsigned(i) / N];
        int iP = (VN == N) ? 0 : i & (N - 1);

        Vf body1_velocityX, body1_velocityY, body1_angularVelocity, body1_lastIterationf;
        Vf body2_velocityX, body2_velocityY, body2_angularVelocity, body2_lastIterationf;

        Vf body1_invMass, body1_invInertia, body1_coords_posX, body1_coords_posY;
        Vf body1_coords_xVectorX, body1_coords_xVectorY, body1_coords_yVectorX, body1_coords_yVectorY;

        Vf body2_invMass, body2_invInertia, body2_coords_posX, body2_coords_posY;
        Vf body2_coords_xVectorX, body2_coords_xVectorY, body2_coords_yVectorX, body2_coords_yVectorY;

        Vf collision_delta1X, collision_delta1Y, collision_delta2X, collision_delta2Y;
        Vf collision_normalX, collision_normalY;
        Vf dummy;

        loadindexed4(
            body1_velocityX, body1_velocityY, body1_angularVelocity, body1_lastIterationf,
            solveBodiesImpulse.data, jointP.body1Index + iP, sizeof(SolveBody));

        loadindexed4(
            body2_velocityX, body2_velocityY, body2_angularVelocity, body2_lastIterationf,
            solveBodiesImpulse.data, jointP.body2Index + iP, sizeof(SolveBody));

        loadindexed8(
            body1_invMass, body1_invInertia, body1_coords_posX, body1_coords_posY,
            body1_coords_xVectorX, body1_coords_xVectorY, body1_coords_yVectorX, body1_coords_yVectorY,
            solveBodiesParams.data, jointP.body1Index + iP, sizeof(SolveBodyParams));

        loadindexed8(
            body2_invMass, body2_invInertia, body2_coords_posX, body2_coords_posY,
            body2_coords_xVectorX, body2_coords_xVectorY, body2_coords_yVectorX, body2_coords_yVectorY,
            solveBodiesParams.data, jointP.body2Index + iP, sizeof(SolveBodyParams));

        loadindexed8(
            collision_delta1X, collision_delta1Y, collision_delta2X, collision_delta2Y,
            collision_normalX, collision_normalY, dummy, dummy,
            contactPoints, jointP.contactPointIndex + iP, sizeof(ContactPoint));

        Vf point1X = collision_delta1X + body1_coords_posX;
        Vf point1Y = collision_delta1Y + body1_coords_posY;
        Vf point2X = collision_delta2X + body2_coords_posX;
        Vf point2Y = collision_delta2Y + body2_coords_posY;

        Vf w1X = collision_delta1X;
        Vf w1Y = collision_delta1Y;
        Vf w2X = point1X - body2_coords_posX;
        Vf w2Y = point1Y - body2_coords_posY;

        // Normal limiter
        RefreshLimiter(jointP.normalLimiter, iP,
            collision_normalX, collision_normalY, -collision_normalX, -collision_normalY,
            w1X, w1Y, w2X, w2Y,
            body1_invMass, body1_invInertia, body2_invMass, body2_invInertia);

        Vf bounce = Vf::zero();
        Vf deltaVelocity = Vf::one(1.f);
        Vf maxPenetrationVelocity = Vf::one(0.1f);
        Vf deltaDepth = Vf::one(1.f);
        Vf errorReduction = Vf::one(0.1f);

        Vf pointVelocity_body1X = (body1_coords_posY - point1Y) * body1_angularVelocity + body1_velocityX;
        Vf pointVelocity_body1Y = (point1X - body1_coords_posX) * body1_angularVelocity + body1_velocityY;

        Vf pointVelocity_body2X = (body2_coords_posY - point2Y) * body2_angularVelocity + body2_velocityX;
        Vf pointVelocity_body2Y = (point2X - body2_coords_posX) * body2_angularVelocity + body2_velocityY;

        Vf relativeVelocityX = pointVelocity_body1X - pointVelocity_body2X;
        Vf relativeVelocityY = pointVelocity_body1Y - pointVelocity_body2Y;

        Vf dv = -bounce * (relativeVelocityX * collision_normalX + relativeVelocityY * collision_normalY);
        Vf depth = (point2X - point1X) * collision_normalX + (point2Y - point1Y) * collision_normalY;

        Vf dstVelocity = max(dv - deltaVelocity, Vf::zero());

        Vf j_normalLimiter_dstVelocity = select(dstVelocity, dstVelocity - maxPenetrationVelocity, depth < deltaDepth);
        Vf j_normalLimiter_dstDisplacingVelocity = errorReduction * max(Vf::zero(), depth - Vf::one(2.0f) * deltaDepth);
        Vf j_normalLimiter_accumulatedDisplacingImpulse = Vf::zero();

        // Friction limiter
        Vf tangentX = -collision_normalY;
        Vf tangentY = collision_normalX;

        RefreshLimiter(jointP.frictionLimiter, iP,
            tangentX, tangentY, -tangentX, -tangentY,
            w1X, w1Y, w2X, w2Y,
            body1_invMass, body1_invInertia, body2_invMass, body2_invInertia);

        store(j_normalLimiter_dstVelocity, &jointP.normalLimiter_dstVelocity[iP]);
        store(j_normalLimiter_dstDisplacingVelocity, &jointP.normalLimiter_dstDisplacingVelocity[iP]);
        store(j_normalLimiter_accumulatedDisplacingImpulse, &jointP.normalLimiter_accumulatedDisplacingImpulse[iP]);
    }
}

template <int VN, int N>
NOINLINE void Solver::PreStepJoints(ContactJointPacked<N>* joint_packed, int jointBegin, int jointEnd)
{
    MICROPROFILE_SCOPEI("Physics", "PreStepJoints", -1);

    typedef simd::VNf<VN> Vf;
    typedef simd::VNi<VN> Vi;
    typedef simd::VNb<VN> Vb;

    assert(jointBegin % VN == 0 && jointEnd % VN == 0);

    for (int jointIndex = jointBegin; jointIndex < jointEnd; jointIndex += VN)
    {
        int i = jointIndex;

        ContactJointPacked<N>& jointP = joint_packed[unsigned(i) / N];
        int iP = (VN == N) ? 0 : i & (N - 1);

        Vf body1_velocityX, body1_velocityY, body1_angularVelocity, body1_lastIterationf;
        Vf body2_velocityX, body2_velocityY, body2_angularVelocity, body2_lastIterationf;

        loadindexed4(body1_velocityX, body1_velocityY, body1_angularVelocity, body1_lastIterationf,
            solveBodiesImpulse.data, jointP.body1Index + iP, sizeof(SolveBody));

        loadindexed4(body2_velocityX, body2_velocityY, body2_angularVelocity, body2_lastIterationf,
            solveBodiesImpulse.data, jointP.body2Index + iP, sizeof(SolveBody));

        Vf j_normalLimiter_compMass1_linearX = Vf::load(&jointP.normalLimiter.compMass1_linearX[iP]);
        Vf j_normalLimiter_compMass1_linearY = Vf::load(&jointP.normalLimiter.compMass1_linearY[iP]);
        Vf j_normalLimiter_compMass2_linearX = Vf::load(&jointP.normalLimiter.compMass2_linearX[iP]);
        Vf j_normalLimiter_compMass2_linearY = Vf::load(&jointP.normalLimiter.compMass2_linearY[iP]);
        Vf j_normalLimiter_compMass1_angular = Vf::load(&jointP.normalLimiter.compMass1_angular[iP]);
        Vf j_normalLimiter_compMass2_angular = Vf::load(&jointP.normalLimiter.compMass2_angular[iP]);
        Vf j_normalLimiter_accumulatedImpulse = Vf::load(&jointP.normalLimiter_accumulatedImpulse[iP]);

        Vf j_frictionLimiter_compMass1_linearX = Vf::load(&jointP.frictionLimiter.compMass1_linearX[iP]);
        Vf j_frictionLimiter_compMass1_linearY = Vf::load(&jointP.frictionLimiter.compMass1_linearY[iP]);
        Vf j_frictionLimiter_compMass2_linearX = Vf::load(&jointP.frictionLimiter.compMass2_linearX[iP]);
        Vf j_frictionLimiter_compMass2_linearY = Vf::load(&jointP.frictionLimiter.compMass2_linearY[iP]);
        Vf j_frictionLimiter_compMass1_angular = Vf::load(&jointP.frictionLimiter.compMass1_angular[iP]);
        Vf j_frictionLimiter_compMass2_angular = Vf::load(&jointP.frictionLimiter.compMass2_angular[iP]);
        Vf j_frictionLimiter_accumulatedImpulse = Vf::load(&jointP.frictionLimiter_accumulatedImpulse[iP]);

        body1_velocityX += j_normalLimiter_compMass1_linearX * j_normalLimiter_accumulatedImpulse;
        body1_velocityY += j_normalLimiter_compMass1_linearY * j_normalLimiter_accumulatedImpulse;
        body1_angularVelocity += j_normalLimiter_compMass1_angular * j_normalLimiter_accumulatedImpulse;

        body2_velocityX += j_normalLimiter_compMass2_linearX * j_normalLimiter_accumulatedImpulse;
        body2_velocityY += j_normalLimiter_compMass2_linearY * j_normalLimiter_accumulatedImpulse;
        body2_angularVelocity += j_normalLimiter_compMass2_angular * j_normalLimiter_accumulatedImpulse;

        body1_velocityX += j_frictionLimiter_compMass1_linearX * j_frictionLimiter_accumulatedImpulse;
        body1_velocityY += j_frictionLimiter_compMass1_linearY * j_frictionLimiter_accumulatedImpulse;
        body1_angularVelocity += j_frictionLimiter_compMass1_angular * j_frictionLimiter_accumulatedImpulse;

        body2_velocityX += j_frictionLimiter_compMass2_linearX * j_frictionLimiter_accumulatedImpulse;
        body2_velocityY += j_frictionLimiter_compMass2_linearY * j_frictionLimiter_accumulatedImpulse;
        body2_angularVelocity += j_frictionLimiter_compMass2_angular * j_frictionLimiter_accumulatedImpulse;

        storeindexed4(body1_velocityX, body1_velocityY, body1_angularVelocity, body1_lastIterationf,
            solveBodiesImpulse.data, jointP.body1Index + iP, sizeof(SolveBody));

        storeindexed4(body2_velocityX, body2_velocityY, body2_angularVelocity, body2_lastIterationf,
            solveBodiesImpulse.data, jointP.body2Index + iP, sizeof(SolveBody));
    }
}

template <int VN, int N>
NOINLINE bool Solver::SolveJointsImpulses(ContactJointPacked<N>* joint_packed, int jointBegin, int jointEnd, int iterationIndex)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsImpulses", -1);

    typedef simd::VNf<VN> Vf;
    typedef simd::VNi<VN> Vi;
    typedef simd::VNb<VN> Vb;

    assert(jointBegin % VN == 0 && jointEnd % VN == 0);

    Vi iterationIndex0 = Vi::one(iterationIndex);
    Vi iterationIndex2 = Vi::one(iterationIndex - 2);

    Vb productive_any = Vb::zero();

    for (int jointIndex = jointBegin; jointIndex < jointEnd; jointIndex += VN)
    {
        int i = jointIndex;

        ContactJointPacked<N>& jointP = joint_packed[unsigned(i) / N];
        int iP = (VN == N) ? 0 : i & (N - 1);

        Vf body1_velocityX, body1_velocityY, body1_angularVelocity, body1_lastIterationf;
        Vf body2_velocityX, body2_velocityY, body2_angularVelocity, body2_lastIterationf;

        loadindexed4(body1_velocityX, body1_velocityY, body1_angularVelocity, body1_lastIterationf,
            solveBodiesImpulse.data, jointP.body1Index + iP, sizeof(SolveBody));

        loadindexed4(body2_velocityX, body2_velocityY, body2_angularVelocity, body2_lastIterationf,
            solveBodiesImpulse.data, jointP.body2Index + iP, sizeof(SolveBody));

        Vi body1_lastIteration = bitcast(body1_lastIterationf);
        Vi body2_lastIteration = bitcast(body2_lastIterationf);

        Vb body1_productive = body1_lastIteration > iterationIndex2;
        Vb body2_productive = body2_lastIteration > iterationIndex2;
        Vb body_productive = body1_productive | body2_productive;

        if (none(body_productive))
            continue;

        Vf j_normalLimiter_normalProjector1X = Vf::load(&jointP.normalLimiter.normalProjector1X[iP]);
        Vf j_normalLimiter_normalProjector1Y = Vf::load(&jointP.normalLimiter.normalProjector1Y[iP]);
        Vf j_normalLimiter_normalProjector2X = Vf::load(&jointP.normalLimiter.normalProjector2X[iP]);
        Vf j_normalLimiter_normalProjector2Y = Vf::load(&jointP.normalLimiter.normalProjector2Y[iP]);
        Vf j_normalLimiter_angularProjector1 = Vf::load(&jointP.normalLimiter.angularProjector1[iP]);
        Vf j_normalLimiter_angularProjector2 = Vf::load(&jointP.normalLimiter.angularProjector2[iP]);

        Vf j_normalLimiter_compMass1_linearX = Vf::load(&jointP.normalLimiter.compMass1_linearX[iP]);
        Vf j_normalLimiter_compMass1_linearY = Vf::load(&jointP.normalLimiter.compMass1_linearY[iP]);
        Vf j_normalLimiter_compMass2_linearX = Vf::load(&jointP.normalLimiter.compMass2_linearX[iP]);
        Vf j_normalLimiter_compMass2_linearY = Vf::load(&jointP.normalLimiter.compMass2_linearY[iP]);
        Vf j_normalLimiter_compMass1_angular = Vf::load(&jointP.normalLimiter.compMass1_angular[iP]);
        Vf j_normalLimiter_compMass2_angular = Vf::load(&jointP.normalLimiter.compMass2_angular[iP]);
        Vf j_normalLimiter_compInvMass = Vf::load(&jointP.normalLimiter.compInvMass[iP]);
        Vf j_normalLimiter_accumulatedImpulse = Vf::load(&jointP.normalLimiter_accumulatedImpulse[iP]);
        Vf j_normalLimiter_dstVelocity = Vf::load(&jointP.normalLimiter_dstVelocity[iP]);

        Vf j_frictionLimiter_normalProjector1X = Vf::load(&jointP.frictionLimiter.normalProjector1X[iP]);
        Vf j_frictionLimiter_normalProjector1Y = Vf::load(&jointP.frictionLimiter.normalProjector1Y[iP]);
        Vf j_frictionLimiter_normalProjector2X = Vf::load(&jointP.frictionLimiter.normalProjector2X[iP]);
        Vf j_frictionLimiter_normalProjector2Y = Vf::load(&jointP.frictionLimiter.normalProjector2Y[iP]);
        Vf j_frictionLimiter_angularProjector1 = Vf::load(&jointP.frictionLimiter.angularProjector1[iP]);
        Vf j_frictionLimiter_angularProjector2 = Vf::load(&jointP.frictionLimiter.angularProjector2[iP]);

        Vf j_frictionLimiter_compMass1_linearX = Vf::load(&jointP.frictionLimiter.compMass1_linearX[iP]);
        Vf j_frictionLimiter_compMass1_linearY = Vf::load(&jointP.frictionLimiter.compMass1_linearY[iP]);
        Vf j_frictionLimiter_compMass2_linearX = Vf::load(&jointP.frictionLimiter.compMass2_linearX[iP]);
        Vf j_frictionLimiter_compMass2_linearY = Vf::load(&jointP.frictionLimiter.compMass2_linearY[iP]);
        Vf j_frictionLimiter_compMass1_angular = Vf::load(&jointP.frictionLimiter.compMass1_angular[iP]);
        Vf j_frictionLimiter_compMass2_angular = Vf::load(&jointP.frictionLimiter.compMass2_angular[iP]);
        Vf j_frictionLimiter_compInvMass = Vf::load(&jointP.frictionLimiter.compInvMass[iP]);
        Vf j_frictionLimiter_accumulatedImpulse = Vf::load(&jointP.frictionLimiter_accumulatedImpulse[iP]);

        Vf normaldV = j_normalLimiter_dstVelocity;

        normaldV -= j_normalLimiter_normalProjector1X * body1_velocityX;
        normaldV -= j_normalLimiter_normalProjector1Y * body1_velocityY;
        normaldV -= j_normalLimiter_angularProjector1 * body1_angularVelocity;

        normaldV -= j_normalLimiter_normalProjector2X * body2_velocityX;
        normaldV -= j_normalLimiter_normalProjector2Y * body2_velocityY;
        normaldV -= j_normalLimiter_angularProjector2 * body2_angularVelocity;

        Vf normalDeltaImpulse = normaldV * j_normalLimiter_compInvMass;

        normalDeltaImpulse = max(normalDeltaImpulse, -j_normalLimiter_accumulatedImpulse);

        body1_velocityX += j_normalLimiter_compMass1_linearX * normalDeltaImpulse;
        body1_velocityY += j_normalLimiter_compMass1_linearY * normalDeltaImpulse;
        body1_angularVelocity += j_normalLimiter_compMass1_angular * normalDeltaImpulse;

        body2_velocityX += j_normalLimiter_compMass2_linearX * normalDeltaImpulse;
        body2_velocityY += j_normalLimiter_compMass2_linearY * normalDeltaImpulse;
        body2_angularVelocity += j_normalLimiter_compMass2_angular * normalDeltaImpulse;

        j_normalLimiter_accumulatedImpulse += normalDeltaImpulse;

        Vf frictiondV = Vf::zero();

        frictiondV -= j_frictionLimiter_normalProjector1X * body1_velocityX;
        frictiondV -= j_frictionLimiter_normalProjector1Y * body1_velocityY;
        frictiondV -= j_frictionLimiter_angularProjector1 * body1_angularVelocity;

        frictiondV -= j_frictionLimiter_normalProjector2X * body2_velocityX;
        frictiondV -= j_frictionLimiter_normalProjector2Y * body2_velocityY;
        frictiondV -= j_frictionLimiter_angularProjector2 * body2_angularVelocity;

        Vf frictionDeltaImpulse = frictiondV * j_frictionLimiter_compInvMass;

        Vf reactionForce = j_normalLimiter_accumulatedImpulse;
        Vf accumulatedImpulse = j_frictionLimiter_accumulatedImpulse;

        Vf frictionForce = accumulatedImpulse + frictionDeltaImpulse;
        Vf reactionForceScaled = reactionForce * Vf::one(kFrictionCoefficient);

        Vf frictionForceAbs = abs(frictionForce);
        Vf reactionForceScaledSigned = flipsign(reactionForceScaled, frictionForce);
        Vf frictionDeltaImpulseAdjusted = reactionForceScaledSigned - accumulatedImpulse;

        frictionDeltaImpulse = select(frictionDeltaImpulse, frictionDeltaImpulseAdjusted, frictionForceAbs > reactionForceScaled);

        j_frictionLimiter_accumulatedImpulse += frictionDeltaImpulse;

        body1_velocityX += j_frictionLimiter_compMass1_linearX * frictionDeltaImpulse;
        body1_velocityY += j_frictionLimiter_compMass1_linearY * frictionDeltaImpulse;
        body1_angularVelocity += j_frictionLimiter_compMass1_angular * frictionDeltaImpulse;

        body2_velocityX += j_frictionLimiter_compMass2_linearX * frictionDeltaImpulse;
        body2_velocityY += j_frictionLimiter_compMass2_linearY * frictionDeltaImpulse;
        body2_angularVelocity += j_frictionLimiter_compMass2_angular * frictionDeltaImpulse;

        store(j_normalLimiter_accumulatedImpulse, &jointP.normalLimiter_accumulatedImpulse[iP]);
        store(j_frictionLimiter_accumulatedImpulse, &jointP.frictionLimiter_accumulatedImpulse[iP]);

        Vf cumulativeImpulse = max(abs(normalDeltaImpulse), abs(frictionDeltaImpulse));

        Vb productive = cumulativeImpulse > Vf::one(kProductiveImpulse);

        productive_any |= productive;

        body1_lastIteration = select(body1_lastIteration, iterationIndex0, productive);
        body2_lastIteration = select(body2_lastIteration, iterationIndex0, productive);

        body1_lastIterationf = bitcast(body1_lastIteration);
        body2_lastIterationf = bitcast(body2_lastIteration);

        storeindexed4(body1_velocityX, body1_velocityY, body1_angularVelocity, body1_lastIterationf,
            solveBodiesImpulse.data, jointP.body1Index + iP, sizeof(SolveBody));

        storeindexed4(body2_velocityX, body2_velocityY, body2_angularVelocity, body2_lastIterationf,
            solveBodiesImpulse.data, jointP.body2Index + iP, sizeof(SolveBody));
    }

    return any(productive_any);
}

template <int VN, int N>
NOINLINE bool Solver::SolveJointsDisplacement(ContactJointPacked<N>* joint_packed, int jointBegin, int jointEnd, int iterationIndex)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsDisplacement", -1);

    typedef simd::VNf<VN> Vf;
    typedef simd::VNi<VN> Vi;
    typedef simd::VNb<VN> Vb;

    assert(jointBegin % VN == 0 && jointEnd % VN == 0);

    Vi iterationIndex0 = Vi::one(iterationIndex);
    Vi iterationIndex2 = Vi::one(iterationIndex - 2);

    Vb productive_any = Vb::zero();

    for (int jointIndex = jointBegin; jointIndex < jointEnd; jointIndex += VN)
    {
        int i = jointIndex;

        ContactJointPacked<N>& jointP = joint_packed[unsigned(i) / N];
        int iP = (VN == N) ? 0 : i & (N - 1);

        Vf body1_velocityX, body1_velocityY, body1_angularVelocity, body1_lastIterationf;
        Vf body2_velocityX, body2_velocityY, body2_angularVelocity, body2_lastIterationf;

        loadindexed4(body1_velocityX, body1_velocityY, body1_angularVelocity, body1_lastIterationf,
            solveBodiesDisplacement.data, jointP.body1Index + iP, sizeof(SolveBody));

        loadindexed4(body2_velocityX, body2_velocityY, body2_angularVelocity, body2_lastIterationf,
            solveBodiesDisplacement.data, jointP.body2Index + iP, sizeof(SolveBody));

        Vi body1_lastIteration = bitcast(body1_lastIterationf);
        Vi body2_lastIteration = bitcast(body2_lastIterationf);

        Vb body1_productive = body1_lastIteration > iterationIndex2;
        Vb body2_productive = body2_lastIteration > iterationIndex2;
        Vb body_productive = body1_productive | body2_productive;

        if (none(body_productive))
            continue;

        Vf j_normalLimiter_normalProjector1X = Vf::load(&jointP.normalLimiter.normalProjector1X[iP]);
        Vf j_normalLimiter_normalProjector1Y = Vf::load(&jointP.normalLimiter.normalProjector1Y[iP]);
        Vf j_normalLimiter_normalProjector2X = Vf::load(&jointP.normalLimiter.normalProjector2X[iP]);
        Vf j_normalLimiter_normalProjector2Y = Vf::load(&jointP.normalLimiter.normalProjector2Y[iP]);
        Vf j_normalLimiter_angularProjector1 = Vf::load(&jointP.normalLimiter.angularProjector1[iP]);
        Vf j_normalLimiter_angularProjector2 = Vf::load(&jointP.normalLimiter.angularProjector2[iP]);

        Vf j_normalLimiter_compMass1_linearX = Vf::load(&jointP.normalLimiter.compMass1_linearX[iP]);
        Vf j_normalLimiter_compMass1_linearY = Vf::load(&jointP.normalLimiter.compMass1_linearY[iP]);
        Vf j_normalLimiter_compMass2_linearX = Vf::load(&jointP.normalLimiter.compMass2_linearX[iP]);
        Vf j_normalLimiter_compMass2_linearY = Vf::load(&jointP.normalLimiter.compMass2_linearY[iP]);
        Vf j_normalLimiter_compMass1_angular = Vf::load(&jointP.normalLimiter.compMass1_angular[iP]);
        Vf j_normalLimiter_compMass2_angular = Vf::load(&jointP.normalLimiter.compMass2_angular[iP]);
        Vf j_normalLimiter_compInvMass = Vf::load(&jointP.normalLimiter.compInvMass[iP]);
        Vf j_normalLimiter_dstDisplacingVelocity = Vf::load(&jointP.normalLimiter_dstDisplacingVelocity[iP]);
        Vf j_normalLimiter_accumulatedDisplacingImpulse = Vf::load(&jointP.normalLimiter_accumulatedDisplacingImpulse[iP]);

        Vf dV = j_normalLimiter_dstDisplacingVelocity;

        dV -= j_normalLimiter_normalProjector1X * body1_velocityX;
        dV -= j_normalLimiter_normalProjector1Y * body1_velocityY;
        dV -= j_normalLimiter_angularProjector1 * body1_angularVelocity;

        dV -= j_normalLimiter_normalProjector2X * body2_velocityX;
        dV -= j_normalLimiter_normalProjector2Y * body2_velocityY;
        dV -= j_normalLimiter_angularProjector2 * body2_angularVelocity;

        Vf displacingDeltaImpulse = dV * j_normalLimiter_compInvMass;

        displacingDeltaImpulse = max(displacingDeltaImpulse, -j_normalLimiter_accumulatedDisplacingImpulse);

        body1_velocityX += j_normalLimiter_compMass1_linearX * displacingDeltaImpulse;
        body1_velocityY += j_normalLimiter_compMass1_linearY * displacingDeltaImpulse;
        body1_angularVelocity += j_normalLimiter_compMass1_angular * displacingDeltaImpulse;

        body2_velocityX += j_normalLimiter_compMass2_linearX * displacingDeltaImpulse;
        body2_velocityY += j_normalLimiter_compMass2_linearY * displacingDeltaImpulse;
        body2_angularVelocity += j_normalLimiter_compMass2_angular * displacingDeltaImpulse;

        j_normalLimiter_accumulatedDisplacingImpulse += displacingDeltaImpulse;

        store(j_normalLimiter_accumulatedDisplacingImpulse, &jointP.normalLimiter_accumulatedDisplacingImpulse[iP]);

        Vb productive = abs(displacingDeltaImpulse) > Vf::one(kProductiveImpulse);

        productive_any |= productive;

        body1_lastIteration = select(body1_lastIteration, iterationIndex0, productive);
        body2_lastIteration = select(body2_lastIteration, iterationIndex0, productive);

        // this is a bit painful :(
        body1_lastIterationf = bitcast(body1_lastIteration);
        body2_lastIterationf = bitcast(body2_lastIteration);

        storeindexed4(body1_velocityX, body1_velocityY, body1_angularVelocity, body1_lastIterationf,
            solveBodiesDisplacement.data, jointP.body1Index + iP, sizeof(SolveBody));

        storeindexed4(body2_velocityX, body2_velocityY, body2_angularVelocity, body2_lastIterationf,
            solveBodiesDisplacement.data, jointP.body2Index + iP, sizeof(SolveBody));
    }

    return any(productive_any);
}