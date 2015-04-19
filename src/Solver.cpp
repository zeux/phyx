#include "Solver.h"

#include "base/Parallel.h"
#include "base/SIMD.h"

const float kProductiveImpulse = 1e-4f;
const float kFrictionCoefficient = 0.3f;

#ifdef __AVX2__
// http://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
#define _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7) \
    do                                                                    \
    {                                                                     \
        __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;            \
        __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;    \
        __t0 = _mm256_unpacklo_ps(row0, row1);                            \
        __t1 = _mm256_unpackhi_ps(row0, row1);                            \
        __t2 = _mm256_unpacklo_ps(row2, row3);                            \
        __t3 = _mm256_unpackhi_ps(row2, row3);                            \
        __t4 = _mm256_unpacklo_ps(row4, row5);                            \
        __t5 = _mm256_unpackhi_ps(row4, row5);                            \
        __t6 = _mm256_unpacklo_ps(row6, row7);                            \
        __t7 = _mm256_unpackhi_ps(row6, row7);                            \
        __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));   \
        __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));   \
        __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));   \
        __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));   \
        __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));   \
        __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));   \
        __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));   \
        __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));   \
        row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);                \
        row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);                \
        row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);                \
        row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);                \
        row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);                \
        row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);                \
        row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);                \
        row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);                \
    } while (0)

inline __m256 _mm256_combine_ps(__m128 a, __m128 b)
{
    return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
}

inline __m256 _mm256_load2_m128(const float* aaddr, const float* baddr)
{
    __m128 a = _mm_load_ps(aaddr);
    __m128 b = _mm_load_ps(baddr);
    return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
}
#endif

NOINLINE void Solver::RefreshJoints(WorkQueue& queue)
{
    MICROPROFILE_SCOPEI("Physics", "RefreshJoints", -1);

    ParallelFor(queue, contactJoints.data(), contactJoints.size(), 8, [](ContactJoint& j, int) { j.Refresh(); });
}

NOINLINE void Solver::PreStepJoints()
{
    MICROPROFILE_SCOPEI("Physics", "PreStepJoints", -1);

    for (auto& joint: contactJoints)
    {
        joint.PreStep();
    }
}

NOINLINE float Solver::SolveJointsAoS(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsAoS", -1);

    SolvePrepareAoS(bodies, bodiesCount);

    {
        MICROPROFILE_SCOPEI("Physics", "Impulse", -1);

        for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
        {
            bool productive = SolveJointsImpulsesAoS(0, contactJoints.size(), iterationIndex);

            if (!productive) break;
        }
    }

    {
        MICROPROFILE_SCOPEI("Physics", "Displacement", -1);

        for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
        {
            bool productive = SolveJointsDisplacementAoS(0, contactJoints.size(), iterationIndex);

            if (!productive) break;
        }
    }

    return SolveFinishAoS();
}

NOINLINE float Solver::SolveJointsSoA_Scalar(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsSoA_Scalar", -1);

    SolvePrepareSoA(joint_packed4, bodies, bodiesCount, 1);

    {
        MICROPROFILE_SCOPEI("Physics", "Impulse", -1);

        for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
        {
            bool productive = SolveJointsImpulsesSoA(joint_packed4.data, 0, contactJoints.size(), iterationIndex);

            if (!productive) break;
        }
    }

    {
        MICROPROFILE_SCOPEI("Physics", "Displacement", -1);

        for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
        {
            bool productive = SolveJointsDisplacementSoA(joint_packed4.data, 0, contactJoints.size(), iterationIndex);

            if (!productive) break;
        }
    }

    return SolveFinishSoA(joint_packed4, bodies, bodiesCount);
}

NOINLINE float Solver::SolveJointsSoA_SSE2(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsSoA_SSE2", -1);

    int groupOffset = SolvePrepareSoA(joint_packed4, bodies, bodiesCount, 4);

    {
        MICROPROFILE_SCOPEI("Physics", "Impulse", -1);

        for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
        {
            bool productive = false;

            productive |= SolveJointsImpulsesSoA_SSE2(joint_packed4.data, 0, groupOffset, iterationIndex);
            productive |= SolveJointsImpulsesSoA(joint_packed4.data, groupOffset, contactJoints.size() - groupOffset, iterationIndex);

            if (!productive) break;
        }
    }

    {
        MICROPROFILE_SCOPEI("Physics", "Displacement", -1);

        for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
        {
            bool productive = false;

            productive |= SolveJointsDisplacementSoA_SSE2(joint_packed4.data, 0, groupOffset, iterationIndex);
            productive |= SolveJointsDisplacementSoA(joint_packed4.data, groupOffset, contactJoints.size() - groupOffset, iterationIndex);

            if (!productive) break;
        }
    }

    return SolveFinishSoA(joint_packed4, bodies, bodiesCount);
}

#ifdef __AVX2__
NOINLINE float Solver::SolveJointsSoA_AVX2(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsSoA_AVX2", -1);

    int groupOffset = SolvePrepareSoA(joint_packed8, bodies, bodiesCount, 8);

    {
        MICROPROFILE_SCOPEI("Physics", "Impulse", -1);

        for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
        {
            bool productive = false;

            productive |= SolveJointsImpulsesSoA_AVX2(joint_packed8.data, 0, groupOffset, iterationIndex);
            productive |= SolveJointsImpulsesSoA(joint_packed8.data, groupOffset, contactJoints.size() - groupOffset, iterationIndex);

            if (!productive) break;
        }
    }

    {
        MICROPROFILE_SCOPEI("Physics", "Displacement", -1);

        for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
        {
            bool productive = false;

            productive |= SolveJointsDisplacementSoA_AVX2(joint_packed8.data, 0, groupOffset, iterationIndex);
            productive |= SolveJointsDisplacementSoA(joint_packed8.data, groupOffset, contactJoints.size() - groupOffset, iterationIndex);

            if (!productive) break;
        }
    }

    return SolveFinishSoA(joint_packed8, bodies, bodiesCount);
}
#endif

#if defined(__AVX2__) && defined(__FMA__)
NOINLINE float Solver::SolveJointsSoA_FMA(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsSoA_FMA", -1);

    int groupOffset = SolvePrepareSoA(joint_packed16, bodies, bodiesCount, 16);

    {
        MICROPROFILE_SCOPEI("Physics", "Impulse", -1);

        for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
        {
            bool productive = false;

            productive |= SolveJointsImpulsesSoA_FMA(joint_packed16.data, 0, groupOffset, iterationIndex);
            productive |= SolveJointsImpulsesSoA(joint_packed16.data, groupOffset, contactJoints.size() - groupOffset, iterationIndex);

            if (!productive) break;
        }
    }

    {
        MICROPROFILE_SCOPEI("Physics", "Displacement", -1);

        for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
        {
            bool productive = false;

            productive |= SolveJointsDisplacementSoA_FMA(joint_packed16.data, 0, groupOffset, iterationIndex);
            productive |= SolveJointsDisplacementSoA(joint_packed16.data, groupOffset, contactJoints.size() - groupOffset, iterationIndex);

            if (!productive) break;
        }
    }

    return SolveFinishSoA(joint_packed16, bodies, bodiesCount);
}
#endif

NOINLINE int Solver::SolvePrepareIndicesSoA(int bodiesCount, int groupSizeTarget)
{
    MICROPROFILE_SCOPEI("Physics", "SolvePrepareIndicesSoA", -1);

    int jointCount = contactJoints.size();

    if (groupSizeTarget == 1)
    {
        for (int i = 0; i < jointCount; ++i)
            joint_index[i] = i;

        return jointCount;
    }
    else
    {
        jointGroup_bodies.resize(bodiesCount);
        jointGroup_joints.resize(jointCount);

        for (int i = 0; i < bodiesCount; ++i)
            jointGroup_bodies[i] = 0;

        for (int i = 0; i < jointCount; ++i)
            jointGroup_joints[i] = i;

        int tag = 0;

        int groupOffset = 0;

        while (jointGroup_joints.size >= groupSizeTarget)
        {
            // gather a group of N joints with non-overlapping bodies
            int groupSize = 0;

            tag++;

            for (int i = 0; i < jointGroup_joints.size && groupSize < groupSizeTarget;)
            {
                int jointIndex = jointGroup_joints[i];
                ContactJoint& joint = contactJoints[jointIndex];

                if (jointGroup_bodies[joint.body1Index] < tag && jointGroup_bodies[joint.body2Index] < tag)
                {
                    jointGroup_bodies[joint.body1Index] = tag;
                    jointGroup_bodies[joint.body2Index] = tag;

                    joint_index[groupOffset + groupSize] = jointIndex;
                    groupSize++;

                    jointGroup_joints[i] = jointGroup_joints[jointGroup_joints.size - 1];
                    jointGroup_joints.size--;
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
        for (int i = 0; i < jointGroup_joints.size; ++i)
            joint_index[groupOffset + i] = jointGroup_joints[i];

        return (groupOffset / groupSizeTarget) * groupSizeTarget;
    }
}

NOINLINE void Solver::SolvePrepareAoS(RigidBody* bodies, int bodiesCount)
{
    MICROPROFILE_SCOPEI("Physics", "SolvePrepareAoS", -1);

    for (int bodyIndex = 0; bodyIndex < bodiesCount; ++bodyIndex)
    {
        bodies[bodyIndex].lastIteration = -1;
        bodies[bodyIndex].lastDisplacementIteration = -1;
    }
}

NOINLINE float Solver::SolveFinishAoS()
{
    MICROPROFILE_SCOPEI("Physics", "SolveFinishAoS", -1);

    int iterationSum = 0;

    for (size_t jointIndex = 0; jointIndex < contactJoints.size(); jointIndex++)
    {
        ContactJoint& joint = contactJoints[jointIndex];

        iterationSum += std::max(joint.body1->lastIteration, joint.body2->lastIteration) + 2;
        iterationSum += std::max(joint.body1->lastDisplacementIteration, joint.body2->lastDisplacementIteration) + 2;
    }

    return float(iterationSum) / float(contactJoints.size());
}

template <int N>
NOINLINE int Solver::SolvePrepareSoA(
    AlignedArray<ContactJointPacked<N>>& joint_packed,
    RigidBody* bodies, int bodiesCount, int groupSizeTarget)
{
    MICROPROFILE_SCOPEI("Physics", "SolvePrepareSoA", -1);

    solveBodiesImpulse.resize(bodiesCount);
    solveBodiesDisplacement.resize(bodiesCount);

    for (int i = 0; i < bodiesCount; ++i)
    {
        solveBodiesImpulse[i].velocity = bodies[i].velocity;
        solveBodiesImpulse[i].angularVelocity = bodies[i].angularVelocity;
        solveBodiesImpulse[i].lastIteration = -1;

        solveBodiesDisplacement[i].velocity = bodies[i].displacingVelocity;
        solveBodiesDisplacement[i].angularVelocity = bodies[i].displacingAngularVelocity;
        solveBodiesDisplacement[i].lastIteration = -1;
    }

    int jointCount = contactJoints.size();

    joint_index.resize(jointCount);

    joint_packed.resize(jointCount);

    int groupOffset = SolvePrepareIndicesSoA(bodiesCount, groupSizeTarget);

    for (int i = 0; i < jointCount; ++i)
    {
        ContactJoint& joint = contactJoints[joint_index[i]];

        ContactJointPacked<N>& jointP = joint_packed[unsigned(i) / N];
        int iP = i & (N - 1);

        jointP.body1Index[iP] = joint.body1Index;
        jointP.body2Index[iP] = joint.body2Index;

        jointP.normalLimiter_normalProjector1X[iP] = joint.normalLimiter.normalProjector1.x;
        jointP.normalLimiter_normalProjector1Y[iP] = joint.normalLimiter.normalProjector1.y;
        jointP.normalLimiter_normalProjector2X[iP] = joint.normalLimiter.normalProjector2.x;
        jointP.normalLimiter_normalProjector2Y[iP] = joint.normalLimiter.normalProjector2.y;
        jointP.normalLimiter_angularProjector1[iP] = joint.normalLimiter.angularProjector1;
        jointP.normalLimiter_angularProjector2[iP] = joint.normalLimiter.angularProjector2;

        jointP.normalLimiter_compMass1_linearX[iP] = joint.normalLimiter.compMass1_linear.x;
        jointP.normalLimiter_compMass1_linearY[iP] = joint.normalLimiter.compMass1_linear.y;
        jointP.normalLimiter_compMass2_linearX[iP] = joint.normalLimiter.compMass2_linear.x;
        jointP.normalLimiter_compMass2_linearY[iP] = joint.normalLimiter.compMass2_linear.y;
        jointP.normalLimiter_compMass1_angular[iP] = joint.normalLimiter.compMass1_angular;
        jointP.normalLimiter_compMass2_angular[iP] = joint.normalLimiter.compMass2_angular;
        jointP.normalLimiter_compInvMass[iP] = joint.normalLimiter.compInvMass;
        jointP.normalLimiter_accumulatedImpulse[iP] = joint.normalLimiter.accumulatedImpulse;

        jointP.normalLimiter_dstVelocity[iP] = joint.normalLimiter.dstVelocity;
        jointP.normalLimiter_dstDisplacingVelocity[iP] = joint.normalLimiter.dstDisplacingVelocity;
        jointP.normalLimiter_accumulatedDisplacingImpulse[iP] = joint.normalLimiter.accumulatedDisplacingImpulse;

        jointP.frictionLimiter_normalProjector1X[iP] = joint.frictionLimiter.normalProjector1.x;
        jointP.frictionLimiter_normalProjector1Y[iP] = joint.frictionLimiter.normalProjector1.y;
        jointP.frictionLimiter_normalProjector2X[iP] = joint.frictionLimiter.normalProjector2.x;
        jointP.frictionLimiter_normalProjector2Y[iP] = joint.frictionLimiter.normalProjector2.y;
        jointP.frictionLimiter_angularProjector1[iP] = joint.frictionLimiter.angularProjector1;
        jointP.frictionLimiter_angularProjector2[iP] = joint.frictionLimiter.angularProjector2;

        jointP.frictionLimiter_compMass1_linearX[iP] = joint.frictionLimiter.compMass1_linear.x;
        jointP.frictionLimiter_compMass1_linearY[iP] = joint.frictionLimiter.compMass1_linear.y;
        jointP.frictionLimiter_compMass2_linearX[iP] = joint.frictionLimiter.compMass2_linear.x;
        jointP.frictionLimiter_compMass2_linearY[iP] = joint.frictionLimiter.compMass2_linear.y;
        jointP.frictionLimiter_compMass1_angular[iP] = joint.frictionLimiter.compMass1_angular;
        jointP.frictionLimiter_compMass2_angular[iP] = joint.frictionLimiter.compMass2_angular;
        jointP.frictionLimiter_compInvMass[iP] = joint.frictionLimiter.compInvMass;
        jointP.frictionLimiter_accumulatedImpulse[iP] = joint.frictionLimiter.accumulatedImpulse;
    }

    return groupOffset;
}

template <int N>
NOINLINE float Solver::SolveFinishSoA(
    AlignedArray<ContactJointPacked<N>>& joint_packed,
    RigidBody* bodies, int bodiesCount)
{
    MICROPROFILE_SCOPEI("Physics", "SolveFinishSoA", -1);

    for (int i = 0; i < bodiesCount; ++i)
    {
        bodies[i].velocity = solveBodiesImpulse[i].velocity;
        bodies[i].angularVelocity = solveBodiesImpulse[i].angularVelocity;

        bodies[i].displacingVelocity = solveBodiesDisplacement[i].velocity;
        bodies[i].displacingAngularVelocity = solveBodiesDisplacement[i].angularVelocity;
    }

    int jointCount = contactJoints.size();

    for (int i = 0; i < jointCount; ++i)
    {
        ContactJoint& joint = contactJoints[joint_index[i]];

        ContactJointPacked<N>& jointP = joint_packed[unsigned(i) / N];
        int iP = i & (N - 1);

        joint.normalLimiter.accumulatedImpulse = jointP.normalLimiter_accumulatedImpulse[iP];
        joint.normalLimiter.accumulatedDisplacingImpulse = jointP.normalLimiter_accumulatedDisplacingImpulse[iP];
        joint.frictionLimiter.accumulatedImpulse = jointP.frictionLimiter_accumulatedImpulse[iP];
    }

    int iterationSum = 0;

    for (int i = 0; i < jointCount; ++i)
    {
        ContactJointPacked<N>& jointP = joint_packed[unsigned(i) / N];
        int iP = i & (N - 1);

        unsigned int bi1 = jointP.body1Index[iP];
        unsigned int bi2 = jointP.body2Index[iP];

        iterationSum += std::max(solveBodiesImpulse[bi1].lastIteration, solveBodiesImpulse[bi2].lastIteration) + 2;
        iterationSum += std::max(solveBodiesDisplacement[bi1].lastIteration, solveBodiesDisplacement[bi2].lastIteration) + 2;
    }

    return float(iterationSum) / float(jointCount);
}

NOINLINE bool Solver::SolveJointsImpulsesAoS(int jointStart, int jointCount, int iterationIndex)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsImpulsesAoS", -1);

    bool productive = false;

    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex++)
    {
        ContactJoint& joint = contactJoints[jointIndex];

        RigidBody* body1 = joint.body1;
        RigidBody* body2 = joint.body2;

        if (body1->lastIteration < iterationIndex - 1 && body2->lastIteration < iterationIndex - 1)
            continue;

        float normaldV = joint.normalLimiter.dstVelocity;

        normaldV -= joint.normalLimiter.normalProjector1.x * body1->velocity.x;
        normaldV -= joint.normalLimiter.normalProjector1.y * body1->velocity.y;
        normaldV -= joint.normalLimiter.angularProjector1 * body1->angularVelocity;

        normaldV -= joint.normalLimiter.normalProjector2.x * body2->velocity.x;
        normaldV -= joint.normalLimiter.normalProjector2.y * body2->velocity.y;
        normaldV -= joint.normalLimiter.angularProjector2 * body2->angularVelocity;

        float normalDeltaImpulse = normaldV * joint.normalLimiter.compInvMass;

        if (normalDeltaImpulse + joint.normalLimiter.accumulatedImpulse < 0.0f)
            normalDeltaImpulse = -joint.normalLimiter.accumulatedImpulse;

        body1->velocity.x += joint.normalLimiter.compMass1_linear.x * normalDeltaImpulse;
        body1->velocity.y += joint.normalLimiter.compMass1_linear.y * normalDeltaImpulse;
        body1->angularVelocity += joint.normalLimiter.compMass1_angular * normalDeltaImpulse;
        body2->velocity.x += joint.normalLimiter.compMass2_linear.x * normalDeltaImpulse;
        body2->velocity.y += joint.normalLimiter.compMass2_linear.y * normalDeltaImpulse;
        body2->angularVelocity += joint.normalLimiter.compMass2_angular * normalDeltaImpulse;

        joint.normalLimiter.accumulatedImpulse += normalDeltaImpulse;

        float frictiondV = 0;

        frictiondV -= joint.frictionLimiter.normalProjector1.x * body1->velocity.x;
        frictiondV -= joint.frictionLimiter.normalProjector1.y * body1->velocity.y;
        frictiondV -= joint.frictionLimiter.angularProjector1 * body1->angularVelocity;

        frictiondV -= joint.frictionLimiter.normalProjector2.x * body2->velocity.x;
        frictiondV -= joint.frictionLimiter.normalProjector2.y * body2->velocity.y;
        frictiondV -= joint.frictionLimiter.angularProjector2 * body2->angularVelocity;

        float frictionDeltaImpulse = frictiondV * joint.frictionLimiter.compInvMass;

        float reactionForce = joint.normalLimiter.accumulatedImpulse;
        float accumulatedImpulse = joint.frictionLimiter.accumulatedImpulse;

        float frictionForce = accumulatedImpulse + frictionDeltaImpulse;
        float frictionCoefficient = kFrictionCoefficient;

        if (fabsf(frictionForce) > (reactionForce * frictionCoefficient))
        {
            float dir = frictionForce > 0.0f ? 1.0f : -1.0f;
            frictionForce = dir * reactionForce * frictionCoefficient;
            frictionDeltaImpulse = frictionForce - accumulatedImpulse;
        }

        joint.frictionLimiter.accumulatedImpulse += frictionDeltaImpulse;

        body1->velocity.x += joint.frictionLimiter.compMass1_linear.x * frictionDeltaImpulse;
        body1->velocity.y += joint.frictionLimiter.compMass1_linear.y * frictionDeltaImpulse;
        body1->angularVelocity += joint.frictionLimiter.compMass1_angular * frictionDeltaImpulse;

        body2->velocity.x += joint.frictionLimiter.compMass2_linear.x * frictionDeltaImpulse;
        body2->velocity.y += joint.frictionLimiter.compMass2_linear.y * frictionDeltaImpulse;
        body2->angularVelocity += joint.frictionLimiter.compMass2_angular * frictionDeltaImpulse;

        float cumulativeImpulse = std::max(fabsf(normalDeltaImpulse), fabsf(frictionDeltaImpulse));

        if (cumulativeImpulse > kProductiveImpulse)
        {
            body1->lastIteration = iterationIndex;
            body2->lastIteration = iterationIndex;
            productive = true;
        }
    }

    return productive;
}

NOINLINE bool Solver::SolveJointsDisplacementAoS(int jointStart, int jointCount, int iterationIndex)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsDisplacementAoS", -1);

    bool productive = false;

    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex++)
    {
        ContactJoint& joint = contactJoints[jointIndex];

        RigidBody* body1 = joint.body1;
        RigidBody* body2 = joint.body2;

        if (body1->lastDisplacementIteration < iterationIndex - 1 && body2->lastDisplacementIteration < iterationIndex - 1)
            continue;

        float dV = joint.normalLimiter.dstDisplacingVelocity;

        dV -= joint.normalLimiter.normalProjector1.x * body1->displacingVelocity.x;
        dV -= joint.normalLimiter.normalProjector1.y * body1->displacingVelocity.y;
        dV -= joint.normalLimiter.angularProjector1 * body1->displacingAngularVelocity;

        dV -= joint.normalLimiter.normalProjector2.x * body2->displacingVelocity.x;
        dV -= joint.normalLimiter.normalProjector2.y * body2->displacingVelocity.y;
        dV -= joint.normalLimiter.angularProjector2 * body2->displacingAngularVelocity;

        float displacingDeltaImpulse = dV * joint.normalLimiter.compInvMass;

        if (displacingDeltaImpulse + joint.normalLimiter.accumulatedDisplacingImpulse < 0.0f)
            displacingDeltaImpulse = -joint.normalLimiter.accumulatedDisplacingImpulse;

        body1->displacingVelocity.x += joint.normalLimiter.compMass1_linear.x * displacingDeltaImpulse;
        body1->displacingVelocity.y += joint.normalLimiter.compMass1_linear.y * displacingDeltaImpulse;
        body1->displacingAngularVelocity += joint.normalLimiter.compMass1_angular * displacingDeltaImpulse;

        body2->displacingVelocity.x += joint.normalLimiter.compMass2_linear.x * displacingDeltaImpulse;
        body2->displacingVelocity.y += joint.normalLimiter.compMass2_linear.y * displacingDeltaImpulse;
        body2->displacingAngularVelocity += joint.normalLimiter.compMass2_angular * displacingDeltaImpulse;

        joint.normalLimiter.accumulatedDisplacingImpulse += displacingDeltaImpulse;

        if (fabsf(displacingDeltaImpulse) > kProductiveImpulse)
        {
            body1->lastDisplacementIteration = iterationIndex;
            body2->lastDisplacementIteration = iterationIndex;
            productive = true;
        }
    }

    return productive;
}

template <int N>
NOINLINE bool Solver::SolveJointsImpulsesSoA(ContactJointPacked<N>* joint_packed, int jointStart, int jointCount, int iterationIndex)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsImpulsesSoA", -1);

    bool productive_any = false;

    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex++)
    {
        int i = jointIndex;

        ContactJointPacked<N>& jointP = joint_packed[unsigned(i) / N];
        int iP = i & (N - 1);

        SolveBody* body1 = &solveBodiesImpulse[jointP.body1Index[iP]];
        SolveBody* body2 = &solveBodiesImpulse[jointP.body2Index[iP]];

        if (body1->lastIteration < iterationIndex - 1 && body2->lastIteration < iterationIndex - 1)
            continue;

        float normaldV = jointP.normalLimiter_dstVelocity[iP];

        normaldV -= jointP.normalLimiter_normalProjector1X[iP] * body1->velocity.x;
        normaldV -= jointP.normalLimiter_normalProjector1Y[iP] * body1->velocity.y;
        normaldV -= jointP.normalLimiter_angularProjector1[iP] * body1->angularVelocity;

        normaldV -= jointP.normalLimiter_normalProjector2X[iP] * body2->velocity.x;
        normaldV -= jointP.normalLimiter_normalProjector2Y[iP] * body2->velocity.y;
        normaldV -= jointP.normalLimiter_angularProjector2[iP] * body2->angularVelocity;

        float normalDeltaImpulse = normaldV * jointP.normalLimiter_compInvMass[iP];

        if (normalDeltaImpulse + jointP.normalLimiter_accumulatedImpulse[iP] < 0.0f)
            normalDeltaImpulse = -jointP.normalLimiter_accumulatedImpulse[iP];

        body1->velocity.x += jointP.normalLimiter_compMass1_linearX[iP] * normalDeltaImpulse;
        body1->velocity.y += jointP.normalLimiter_compMass1_linearY[iP] * normalDeltaImpulse;
        body1->angularVelocity += jointP.normalLimiter_compMass1_angular[iP] * normalDeltaImpulse;
        body2->velocity.x += jointP.normalLimiter_compMass2_linearX[iP] * normalDeltaImpulse;
        body2->velocity.y += jointP.normalLimiter_compMass2_linearY[iP] * normalDeltaImpulse;
        body2->angularVelocity += jointP.normalLimiter_compMass2_angular[iP] * normalDeltaImpulse;

        jointP.normalLimiter_accumulatedImpulse[iP] += normalDeltaImpulse;

        float frictiondV = 0;

        frictiondV -= jointP.frictionLimiter_normalProjector1X[iP] * body1->velocity.x;
        frictiondV -= jointP.frictionLimiter_normalProjector1Y[iP] * body1->velocity.y;
        frictiondV -= jointP.frictionLimiter_angularProjector1[iP] * body1->angularVelocity;

        frictiondV -= jointP.frictionLimiter_normalProjector2X[iP] * body2->velocity.x;
        frictiondV -= jointP.frictionLimiter_normalProjector2Y[iP] * body2->velocity.y;
        frictiondV -= jointP.frictionLimiter_angularProjector2[iP] * body2->angularVelocity;

        float frictionDeltaImpulse = frictiondV * jointP.frictionLimiter_compInvMass[iP];

        float reactionForce = jointP.normalLimiter_accumulatedImpulse[iP];
        float accumulatedImpulse = jointP.frictionLimiter_accumulatedImpulse[iP];

        float frictionForce = accumulatedImpulse + frictionDeltaImpulse;
        float frictionCoefficient = kFrictionCoefficient;

        if (fabsf(frictionForce) > (reactionForce * frictionCoefficient))
        {
            float dir = frictionForce > 0.0f ? 1.0f : -1.0f;
            frictionForce = dir * reactionForce * frictionCoefficient;
            frictionDeltaImpulse = frictionForce - accumulatedImpulse;
        }

        jointP.frictionLimiter_accumulatedImpulse[iP] += frictionDeltaImpulse;

        body1->velocity.x += jointP.frictionLimiter_compMass1_linearX[iP] * frictionDeltaImpulse;
        body1->velocity.y += jointP.frictionLimiter_compMass1_linearY[iP] * frictionDeltaImpulse;
        body1->angularVelocity += jointP.frictionLimiter_compMass1_angular[iP] * frictionDeltaImpulse;

        body2->velocity.x += jointP.frictionLimiter_compMass2_linearX[iP] * frictionDeltaImpulse;
        body2->velocity.y += jointP.frictionLimiter_compMass2_linearY[iP] * frictionDeltaImpulse;
        body2->angularVelocity += jointP.frictionLimiter_compMass2_angular[iP] * frictionDeltaImpulse;

        float cumulativeImpulse = std::max(fabsf(normalDeltaImpulse), fabsf(frictionDeltaImpulse));

        if (cumulativeImpulse > kProductiveImpulse)
        {
            body1->lastIteration = iterationIndex;
            body2->lastIteration = iterationIndex;
            productive_any = true;
        }
    }

    return productive_any;
}

NOINLINE bool Solver::SolveJointsImpulsesSoA_SSE2(ContactJointPacked<4>* joint_packed, int jointStart, int jointCount, int iterationIndex)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsImpulsesSoA_SSE2", -1);

    assert(jointStart % 4 == 0 && jointCount % 4 == 0);

    V4i iterationIndex0 = V4i::one(iterationIndex);
    V4i iterationIndex2 = V4i::one(iterationIndex - 2);

    V4b productive_any = V4b::zero();

    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 4)
    {
        int i = jointIndex;

        ContactJointPacked<4>& jointP = joint_packed[i >> 2];
        int iP = 0;

        V4f row0, row1, row2, row3;

        static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Loading assumes fixed layout");

        row0 = _mm_load_ps(&solveBodiesImpulse[jointP.body1Index[iP + 0]].velocity.x);
        row1 = _mm_load_ps(&solveBodiesImpulse[jointP.body1Index[iP + 1]].velocity.x);
        row2 = _mm_load_ps(&solveBodiesImpulse[jointP.body1Index[iP + 2]].velocity.x);
        row3 = _mm_load_ps(&solveBodiesImpulse[jointP.body1Index[iP + 3]].velocity.x);

        _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

        V4f body1_velocityX = row0;
        V4f body1_velocityY = row1;
        V4f body1_angularVelocity = row2;
        V4i body1_lastIteration = bitcast(row3);

        row0 = _mm_load_ps(&solveBodiesImpulse[jointP.body2Index[iP + 0]].velocity.x);
        row1 = _mm_load_ps(&solveBodiesImpulse[jointP.body2Index[iP + 1]].velocity.x);
        row2 = _mm_load_ps(&solveBodiesImpulse[jointP.body2Index[iP + 2]].velocity.x);
        row3 = _mm_load_ps(&solveBodiesImpulse[jointP.body2Index[iP + 3]].velocity.x);

        _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

        V4f body2_velocityX = row0;
        V4f body2_velocityY = row1;
        V4f body2_angularVelocity = row2;
        V4i body2_lastIteration = bitcast(row3);

        V4b body1_productive = body1_lastIteration > iterationIndex2;
        V4b body2_productive = body2_lastIteration > iterationIndex2;
        V4b body_productive = body1_productive | body2_productive;

        if (none(body_productive))
            continue;

        V4f j_normalLimiter_normalProjector1X = _mm_load_ps(&jointP.normalLimiter_normalProjector1X[iP]);
        V4f j_normalLimiter_normalProjector1Y = _mm_load_ps(&jointP.normalLimiter_normalProjector1Y[iP]);
        V4f j_normalLimiter_normalProjector2X = _mm_load_ps(&jointP.normalLimiter_normalProjector2X[iP]);
        V4f j_normalLimiter_normalProjector2Y = _mm_load_ps(&jointP.normalLimiter_normalProjector2Y[iP]);
        V4f j_normalLimiter_angularProjector1 = _mm_load_ps(&jointP.normalLimiter_angularProjector1[iP]);
        V4f j_normalLimiter_angularProjector2 = _mm_load_ps(&jointP.normalLimiter_angularProjector2[iP]);

        V4f j_normalLimiter_compMass1_linearX = _mm_load_ps(&jointP.normalLimiter_compMass1_linearX[iP]);
        V4f j_normalLimiter_compMass1_linearY = _mm_load_ps(&jointP.normalLimiter_compMass1_linearY[iP]);
        V4f j_normalLimiter_compMass2_linearX = _mm_load_ps(&jointP.normalLimiter_compMass2_linearX[iP]);
        V4f j_normalLimiter_compMass2_linearY = _mm_load_ps(&jointP.normalLimiter_compMass2_linearY[iP]);
        V4f j_normalLimiter_compMass1_angular = _mm_load_ps(&jointP.normalLimiter_compMass1_angular[iP]);
        V4f j_normalLimiter_compMass2_angular = _mm_load_ps(&jointP.normalLimiter_compMass2_angular[iP]);
        V4f j_normalLimiter_compInvMass = _mm_load_ps(&jointP.normalLimiter_compInvMass[iP]);
        V4f j_normalLimiter_accumulatedImpulse = _mm_load_ps(&jointP.normalLimiter_accumulatedImpulse[iP]);
        V4f j_normalLimiter_dstVelocity = _mm_load_ps(&jointP.normalLimiter_dstVelocity[iP]);

        V4f j_frictionLimiter_normalProjector1X = _mm_load_ps(&jointP.frictionLimiter_normalProjector1X[iP]);
        V4f j_frictionLimiter_normalProjector1Y = _mm_load_ps(&jointP.frictionLimiter_normalProjector1Y[iP]);
        V4f j_frictionLimiter_normalProjector2X = _mm_load_ps(&jointP.frictionLimiter_normalProjector2X[iP]);
        V4f j_frictionLimiter_normalProjector2Y = _mm_load_ps(&jointP.frictionLimiter_normalProjector2Y[iP]);
        V4f j_frictionLimiter_angularProjector1 = _mm_load_ps(&jointP.frictionLimiter_angularProjector1[iP]);
        V4f j_frictionLimiter_angularProjector2 = _mm_load_ps(&jointP.frictionLimiter_angularProjector2[iP]);

        V4f j_frictionLimiter_compMass1_linearX = _mm_load_ps(&jointP.frictionLimiter_compMass1_linearX[iP]);
        V4f j_frictionLimiter_compMass1_linearY = _mm_load_ps(&jointP.frictionLimiter_compMass1_linearY[iP]);
        V4f j_frictionLimiter_compMass2_linearX = _mm_load_ps(&jointP.frictionLimiter_compMass2_linearX[iP]);
        V4f j_frictionLimiter_compMass2_linearY = _mm_load_ps(&jointP.frictionLimiter_compMass2_linearY[iP]);
        V4f j_frictionLimiter_compMass1_angular = _mm_load_ps(&jointP.frictionLimiter_compMass1_angular[iP]);
        V4f j_frictionLimiter_compMass2_angular = _mm_load_ps(&jointP.frictionLimiter_compMass2_angular[iP]);
        V4f j_frictionLimiter_compInvMass = _mm_load_ps(&jointP.frictionLimiter_compInvMass[iP]);
        V4f j_frictionLimiter_accumulatedImpulse = _mm_load_ps(&jointP.frictionLimiter_accumulatedImpulse[iP]);

        V4f normaldV = j_normalLimiter_dstVelocity;

        normaldV -= j_normalLimiter_normalProjector1X * body1_velocityX;
        normaldV -= j_normalLimiter_normalProjector1Y * body1_velocityY;
        normaldV -= j_normalLimiter_angularProjector1 * body1_angularVelocity;

        normaldV -= j_normalLimiter_normalProjector2X * body2_velocityX;
        normaldV -= j_normalLimiter_normalProjector2Y * body2_velocityY;
        normaldV -= j_normalLimiter_angularProjector2 * body2_angularVelocity;

        V4f normalDeltaImpulse = normaldV * j_normalLimiter_compInvMass;

        normalDeltaImpulse = max(normalDeltaImpulse, -j_normalLimiter_accumulatedImpulse);

        body1_velocityX += j_normalLimiter_compMass1_linearX * normalDeltaImpulse;
        body1_velocityY += j_normalLimiter_compMass1_linearY * normalDeltaImpulse;
        body1_angularVelocity += j_normalLimiter_compMass1_angular * normalDeltaImpulse;

        body2_velocityX += j_normalLimiter_compMass2_linearX * normalDeltaImpulse;
        body2_velocityY += j_normalLimiter_compMass2_linearY * normalDeltaImpulse;
        body2_angularVelocity += j_normalLimiter_compMass2_angular * normalDeltaImpulse;

        j_normalLimiter_accumulatedImpulse += normalDeltaImpulse;

        V4f frictiondV = V4f::zero();

        frictiondV -= j_frictionLimiter_normalProjector1X * body1_velocityX;
        frictiondV -= j_frictionLimiter_normalProjector1Y * body1_velocityY;
        frictiondV -= j_frictionLimiter_angularProjector1 * body1_angularVelocity;

        frictiondV -= j_frictionLimiter_normalProjector2X * body2_velocityX;
        frictiondV -= j_frictionLimiter_normalProjector2Y * body2_velocityY;
        frictiondV -= j_frictionLimiter_angularProjector2 * body2_angularVelocity;

        V4f frictionDeltaImpulse = frictiondV * j_frictionLimiter_compInvMass;

        V4f reactionForce = j_normalLimiter_accumulatedImpulse;
        V4f accumulatedImpulse = j_frictionLimiter_accumulatedImpulse;

        V4f frictionForce = accumulatedImpulse + frictionDeltaImpulse;
        V4f reactionForceScaled = reactionForce * V4f::one(kFrictionCoefficient);

        V4f frictionForceAbs = abs(frictionForce);
        V4f reactionForceScaledSigned = flipsign(reactionForceScaled, frictionForce);
        V4f frictionDeltaImpulseAdjusted = reactionForceScaledSigned - accumulatedImpulse;

        frictionDeltaImpulse = select(frictionDeltaImpulse, frictionDeltaImpulseAdjusted, frictionForceAbs > reactionForceScaled);

        j_frictionLimiter_accumulatedImpulse += frictionDeltaImpulse;

        body1_velocityX += j_frictionLimiter_compMass1_linearX * frictionDeltaImpulse;
        body1_velocityY += j_frictionLimiter_compMass1_linearY * frictionDeltaImpulse;
        body1_angularVelocity += j_frictionLimiter_compMass1_angular * frictionDeltaImpulse;

        body2_velocityX += j_frictionLimiter_compMass2_linearX * frictionDeltaImpulse;
        body2_velocityY += j_frictionLimiter_compMass2_linearY * frictionDeltaImpulse;
        body2_angularVelocity += j_frictionLimiter_compMass2_angular * frictionDeltaImpulse;

        _mm_store_ps(&jointP.normalLimiter_accumulatedImpulse[iP], j_normalLimiter_accumulatedImpulse);
        _mm_store_ps(&jointP.frictionLimiter_accumulatedImpulse[iP], j_frictionLimiter_accumulatedImpulse);

        V4f cumulativeImpulse = max(abs(normalDeltaImpulse), abs(frictionDeltaImpulse));

        V4b productive = cumulativeImpulse > V4f::one(kProductiveImpulse);

        productive_any |= productive;

        body1_lastIteration = select(body1_lastIteration, iterationIndex0, productive);
        body2_lastIteration = select(body2_lastIteration, iterationIndex0, productive);

        // this is a bit painful :(
        static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Storing assumes fixed layout");

        row0 = body1_velocityX;
        row1 = body1_velocityY;
        row2 = body1_angularVelocity;
        row3 = bitcast(body1_lastIteration);

        _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP + 0]].velocity.x, row0);
        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP + 1]].velocity.x, row1);
        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP + 2]].velocity.x, row2);
        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP + 3]].velocity.x, row3);

        row0 = body2_velocityX;
        row1 = body2_velocityY;
        row2 = body2_angularVelocity;
        row3 = bitcast(body2_lastIteration);

        _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP + 0]].velocity.x, row0);
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP + 1]].velocity.x, row1);
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP + 2]].velocity.x, row2);
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP + 3]].velocity.x, row3);
    }

    return any(productive_any);
}

#ifdef __AVX2__
NOINLINE bool Solver::SolveJointsImpulsesSoA_AVX2(ContactJointPacked<8>* joint_packed, int jointStart, int jointCount, int iterationIndex)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsImpulsesSoA_AVX2", -1);

    assert(jointStart % 8 == 0 && jointCount % 8 == 0);

    V8f sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    V8i iterationIndex0 = _mm256_set1_epi32(iterationIndex);
    V8i iterationIndex2 = _mm256_set1_epi32(iterationIndex - 2);

    V8i productive_any = _mm256_setzero_si256();

    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 8)
    {
        int i = jointIndex;

        ContactJointPacked<8>& jointP = joint_packed[i >> 3];
        int iP = 0;

        V8f row0, row1, row2, row3, row4, row5, row6, row7;

        static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Loading assumes fixed layout");

        row0 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP + 0]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP + 0]].velocity.x);
        row1 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP + 1]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP + 1]].velocity.x);
        row2 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP + 2]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP + 2]].velocity.x);
        row3 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP + 3]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP + 3]].velocity.x);
        row4 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP + 4]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP + 4]].velocity.x);
        row5 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP + 5]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP + 5]].velocity.x);
        row6 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP + 6]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP + 6]].velocity.x);
        row7 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP + 7]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP + 7]].velocity.x);

        _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

        V8f body1_velocityX = row0;
        V8f body1_velocityY = row1;
        V8f body1_angularVelocity = row2;
        V8i body1_lastIteration = bitcast(row3);

        V8f body2_velocityX = row4;
        V8f body2_velocityY = row5;
        V8f body2_angularVelocity = row6;
        V8i body2_lastIteration = bitcast(row7);

        V8i body_lastIteration = _mm256_max_epi32(body1_lastIteration, body2_lastIteration);
        V8i body_productive = _mm256_cmpgt_epi32(body_lastIteration, iterationIndex2);

        if (_mm256_movemask_epi8(body_productive) == 0)
            continue;

        V8f j_normalLimiter_normalProjector1X = _mm256_load_ps(&jointP.normalLimiter_normalProjector1X[iP]);
        V8f j_normalLimiter_normalProjector1Y = _mm256_load_ps(&jointP.normalLimiter_normalProjector1Y[iP]);
        V8f j_normalLimiter_normalProjector2X = _mm256_load_ps(&jointP.normalLimiter_normalProjector2X[iP]);
        V8f j_normalLimiter_normalProjector2Y = _mm256_load_ps(&jointP.normalLimiter_normalProjector2Y[iP]);
        V8f j_normalLimiter_angularProjector1 = _mm256_load_ps(&jointP.normalLimiter_angularProjector1[iP]);
        V8f j_normalLimiter_angularProjector2 = _mm256_load_ps(&jointP.normalLimiter_angularProjector2[iP]);

        V8f j_normalLimiter_compMass1_linearX = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearX[iP]);
        V8f j_normalLimiter_compMass1_linearY = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearY[iP]);
        V8f j_normalLimiter_compMass2_linearX = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearX[iP]);
        V8f j_normalLimiter_compMass2_linearY = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearY[iP]);
        V8f j_normalLimiter_compMass1_angular = _mm256_load_ps(&jointP.normalLimiter_compMass1_angular[iP]);
        V8f j_normalLimiter_compMass2_angular = _mm256_load_ps(&jointP.normalLimiter_compMass2_angular[iP]);
        V8f j_normalLimiter_compInvMass = _mm256_load_ps(&jointP.normalLimiter_compInvMass[iP]);
        V8f j_normalLimiter_accumulatedImpulse = _mm256_load_ps(&jointP.normalLimiter_accumulatedImpulse[iP]);
        V8f j_normalLimiter_dstVelocity = _mm256_load_ps(&jointP.normalLimiter_dstVelocity[iP]);

        V8f j_frictionLimiter_normalProjector1X = _mm256_load_ps(&jointP.frictionLimiter_normalProjector1X[iP]);
        V8f j_frictionLimiter_normalProjector1Y = _mm256_load_ps(&jointP.frictionLimiter_normalProjector1Y[iP]);
        V8f j_frictionLimiter_normalProjector2X = _mm256_load_ps(&jointP.frictionLimiter_normalProjector2X[iP]);
        V8f j_frictionLimiter_normalProjector2Y = _mm256_load_ps(&jointP.frictionLimiter_normalProjector2Y[iP]);
        V8f j_frictionLimiter_angularProjector1 = _mm256_load_ps(&jointP.frictionLimiter_angularProjector1[iP]);
        V8f j_frictionLimiter_angularProjector2 = _mm256_load_ps(&jointP.frictionLimiter_angularProjector2[iP]);

        V8f j_frictionLimiter_compMass1_linearX = _mm256_load_ps(&jointP.frictionLimiter_compMass1_linearX[iP]);
        V8f j_frictionLimiter_compMass1_linearY = _mm256_load_ps(&jointP.frictionLimiter_compMass1_linearY[iP]);
        V8f j_frictionLimiter_compMass2_linearX = _mm256_load_ps(&jointP.frictionLimiter_compMass2_linearX[iP]);
        V8f j_frictionLimiter_compMass2_linearY = _mm256_load_ps(&jointP.frictionLimiter_compMass2_linearY[iP]);
        V8f j_frictionLimiter_compMass1_angular = _mm256_load_ps(&jointP.frictionLimiter_compMass1_angular[iP]);
        V8f j_frictionLimiter_compMass2_angular = _mm256_load_ps(&jointP.frictionLimiter_compMass2_angular[iP]);
        V8f j_frictionLimiter_compInvMass = _mm256_load_ps(&jointP.frictionLimiter_compInvMass[iP]);
        V8f j_frictionLimiter_accumulatedImpulse = _mm256_load_ps(&jointP.frictionLimiter_accumulatedImpulse[iP]);

        V8f normaldV = j_normalLimiter_dstVelocity;

        normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector1X, body1_velocityX));
        normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector1Y, body1_velocityY));
        normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_angularProjector1, body1_angularVelocity));

        normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector2X, body2_velocityX));
        normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector2Y, body2_velocityY));
        normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_angularProjector2, body2_angularVelocity));

        V8f normalDeltaImpulse = _mm256_mul_ps(normaldV, j_normalLimiter_compInvMass);

        normalDeltaImpulse = _mm256_max_ps(normalDeltaImpulse, _mm256_xor_ps(sign, j_normalLimiter_accumulatedImpulse));

        body1_velocityX = _mm256_add_ps(body1_velocityX, _mm256_mul_ps(j_normalLimiter_compMass1_linearX, normalDeltaImpulse));
        body1_velocityY = _mm256_add_ps(body1_velocityY, _mm256_mul_ps(j_normalLimiter_compMass1_linearY, normalDeltaImpulse));
        body1_angularVelocity = _mm256_add_ps(body1_angularVelocity, _mm256_mul_ps(j_normalLimiter_compMass1_angular, normalDeltaImpulse));

        body2_velocityX = _mm256_add_ps(body2_velocityX, _mm256_mul_ps(j_normalLimiter_compMass2_linearX, normalDeltaImpulse));
        body2_velocityY = _mm256_add_ps(body2_velocityY, _mm256_mul_ps(j_normalLimiter_compMass2_linearY, normalDeltaImpulse));
        body2_angularVelocity = _mm256_add_ps(body2_angularVelocity, _mm256_mul_ps(j_normalLimiter_compMass2_angular, normalDeltaImpulse));

        j_normalLimiter_accumulatedImpulse = _mm256_add_ps(j_normalLimiter_accumulatedImpulse, normalDeltaImpulse);

        V8f frictiondV = V8f::zero();

        frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_normalProjector1X, body1_velocityX));
        frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_normalProjector1Y, body1_velocityY));
        frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_angularProjector1, body1_angularVelocity));

        frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_normalProjector2X, body2_velocityX));
        frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_normalProjector2Y, body2_velocityY));
        frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_angularProjector2, body2_angularVelocity));

        V8f frictionDeltaImpulse = _mm256_mul_ps(frictiondV, j_frictionLimiter_compInvMass);

        V8f reactionForce = j_normalLimiter_accumulatedImpulse;
        V8f accumulatedImpulse = j_frictionLimiter_accumulatedImpulse;

        V8f frictionForce = _mm256_add_ps(accumulatedImpulse, frictionDeltaImpulse);
        V8f reactionForceScaled = _mm256_mul_ps(reactionForce, _mm256_set1_ps(kFrictionCoefficient));

        V8f frictionForceAbs = _mm256_andnot_ps(sign, frictionForce);
        V8f reactionForceScaledSigned = _mm256_xor_ps(_mm256_and_ps(frictionForce, sign), reactionForceScaled);
        V8f frictionDeltaImpulseAdjusted = _mm256_sub_ps(reactionForceScaledSigned, accumulatedImpulse);

        V8f frictionSelector = _mm256_cmp_ps(frictionForceAbs, reactionForceScaled, _CMP_GT_OQ);

        frictionDeltaImpulse = _mm256_blendv_ps(frictionDeltaImpulse, frictionDeltaImpulseAdjusted, frictionSelector);

        j_frictionLimiter_accumulatedImpulse = _mm256_add_ps(j_frictionLimiter_accumulatedImpulse, frictionDeltaImpulse);

        body1_velocityX = _mm256_add_ps(body1_velocityX, _mm256_mul_ps(j_frictionLimiter_compMass1_linearX, frictionDeltaImpulse));
        body1_velocityY = _mm256_add_ps(body1_velocityY, _mm256_mul_ps(j_frictionLimiter_compMass1_linearY, frictionDeltaImpulse));
        body1_angularVelocity = _mm256_add_ps(body1_angularVelocity, _mm256_mul_ps(j_frictionLimiter_compMass1_angular, frictionDeltaImpulse));

        body2_velocityX = _mm256_add_ps(body2_velocityX, _mm256_mul_ps(j_frictionLimiter_compMass2_linearX, frictionDeltaImpulse));
        body2_velocityY = _mm256_add_ps(body2_velocityY, _mm256_mul_ps(j_frictionLimiter_compMass2_linearY, frictionDeltaImpulse));
        body2_angularVelocity = _mm256_add_ps(body2_angularVelocity, _mm256_mul_ps(j_frictionLimiter_compMass2_angular, frictionDeltaImpulse));

        _mm256_store_ps(&jointP.normalLimiter_accumulatedImpulse[iP], j_normalLimiter_accumulatedImpulse);
        _mm256_store_ps(&jointP.frictionLimiter_accumulatedImpulse[iP], j_frictionLimiter_accumulatedImpulse);

        V8f cumulativeImpulse = _mm256_max_ps(_mm256_andnot_ps(sign, normalDeltaImpulse), _mm256_andnot_ps(sign, frictionDeltaImpulse));

        V8i productive = bitcast(V8f(_mm256_cmp_ps(cumulativeImpulse, _mm256_set1_ps(kProductiveImpulse), _CMP_GT_OQ)));

        productive_any = _mm256_or_si256(productive_any, productive);

        body1_lastIteration = _mm256_blendv_epi8(body1_lastIteration, iterationIndex0, productive);
        body2_lastIteration = _mm256_blendv_epi8(body2_lastIteration, iterationIndex0, productive);

        // this is a bit painful :(
        static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Storing assumes fixed layout");

        row0 = body1_velocityX;
        row1 = body1_velocityY;
        row2 = body1_angularVelocity;
        row3 = bitcast(body1_lastIteration);

        row4 = body2_velocityX;
        row5 = body2_velocityY;
        row6 = body2_angularVelocity;
        row7 = bitcast(body2_lastIteration);

        _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP + 0]].velocity.x, _mm256_extractf128_ps(row0, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP + 0]].velocity.x, _mm256_extractf128_ps(row0, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP + 1]].velocity.x, _mm256_extractf128_ps(row1, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP + 1]].velocity.x, _mm256_extractf128_ps(row1, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP + 2]].velocity.x, _mm256_extractf128_ps(row2, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP + 2]].velocity.x, _mm256_extractf128_ps(row2, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP + 3]].velocity.x, _mm256_extractf128_ps(row3, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP + 3]].velocity.x, _mm256_extractf128_ps(row3, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP + 4]].velocity.x, _mm256_extractf128_ps(row4, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP + 4]].velocity.x, _mm256_extractf128_ps(row4, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP + 5]].velocity.x, _mm256_extractf128_ps(row5, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP + 5]].velocity.x, _mm256_extractf128_ps(row5, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP + 6]].velocity.x, _mm256_extractf128_ps(row6, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP + 6]].velocity.x, _mm256_extractf128_ps(row6, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP + 7]].velocity.x, _mm256_extractf128_ps(row7, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP + 7]].velocity.x, _mm256_extractf128_ps(row7, 1));
    }

    return _mm256_movemask_epi8(productive_any) != 0;
}
#endif

#if defined(__AVX2__) && defined(__FMA__)
NOINLINE bool Solver::SolveJointsImpulsesSoA_FMA(ContactJointPacked<16>* joint_packed, int jointStart, int jointCount, int iterationIndex)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsImpulsesSoA_FMA", -1);

    assert(jointStart % 16 == 0 && jointCount % 16 == 0);

    V8f sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    V8i iterationIndex0 = _mm256_set1_epi32(iterationIndex);
    V8i iterationIndex2 = _mm256_set1_epi32(iterationIndex - 2);

    V8i productive_any = _mm256_setzero_si256();

    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 16)
    {
        int i = jointIndex;

        ContactJointPacked<16>& jointP = joint_packed[i >> 4];
        int iP_0 = 0;
        int iP_1 = 8;

        V8f row0, row1, row2, row3, row4, row5, row6, row7;

        static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Loading assumes fixed layout");

        row0 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_0 + 0]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_0 + 0]].velocity.x);
        row1 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_0 + 1]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_0 + 1]].velocity.x);
        row2 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_0 + 2]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_0 + 2]].velocity.x);
        row3 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_0 + 3]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_0 + 3]].velocity.x);
        row4 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_0 + 4]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_0 + 4]].velocity.x);
        row5 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_0 + 5]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_0 + 5]].velocity.x);
        row6 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_0 + 6]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_0 + 6]].velocity.x);
        row7 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_0 + 7]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_0 + 7]].velocity.x);

        _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

        V8f body1_velocityX_0 = row0;
        V8f body1_velocityY_0 = row1;
        V8f body1_angularVelocity_0 = row2;
        V8i body1_lastIteration_0 = bitcast(row3);

        V8f body2_velocityX_0 = row4;
        V8f body2_velocityY_0 = row5;
        V8f body2_angularVelocity_0 = row6;
        V8i body2_lastIteration_0 = bitcast(row7);

        row0 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_1 + 0]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_1 + 0]].velocity.x);
        row1 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_1 + 1]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_1 + 1]].velocity.x);
        row2 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_1 + 2]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_1 + 2]].velocity.x);
        row3 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_1 + 3]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_1 + 3]].velocity.x);
        row4 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_1 + 4]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_1 + 4]].velocity.x);
        row5 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_1 + 5]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_1 + 5]].velocity.x);
        row6 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_1 + 6]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_1 + 6]].velocity.x);
        row7 = _mm256_load2_m128(&solveBodiesImpulse[jointP.body1Index[iP_1 + 7]].velocity.x, &solveBodiesImpulse[jointP.body2Index[iP_1 + 7]].velocity.x);

        _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

        V8f body1_velocityX_1 = row0;
        V8f body1_velocityY_1 = row1;
        V8f body1_angularVelocity_1 = row2;
        V8i body1_lastIteration_1 = bitcast(row3);

        V8f body2_velocityX_1 = row4;
        V8f body2_velocityY_1 = row5;
        V8f body2_angularVelocity_1 = row6;
        V8i body2_lastIteration_1 = bitcast(row7);

        V8i body_lastIteration_0 = _mm256_max_epi32(body1_lastIteration_0, body2_lastIteration_0);
        V8i body_lastIteration_1 = _mm256_max_epi32(body1_lastIteration_1, body2_lastIteration_1);

        V8i body_productive_0 = _mm256_cmpgt_epi32(body_lastIteration_0, iterationIndex2);
        V8i body_productive_1 = _mm256_cmpgt_epi32(body_lastIteration_1, iterationIndex2);
        V8i body_productive = _mm256_or_si256(body_productive_0, body_productive_1);

        if (_mm256_movemask_epi8(body_productive) == 0)
            continue;

        V8f j_normalLimiter_normalProjector1X_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1X[iP_0]);
        V8f j_normalLimiter_normalProjector1Y_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1Y[iP_0]);
        V8f j_normalLimiter_normalProjector2X_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2X[iP_0]);
        V8f j_normalLimiter_normalProjector2Y_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2Y[iP_0]);
        V8f j_normalLimiter_angularProjector1_0 = _mm256_load_ps(&jointP.normalLimiter_angularProjector1[iP_0]);
        V8f j_normalLimiter_angularProjector2_0 = _mm256_load_ps(&jointP.normalLimiter_angularProjector2[iP_0]);

        V8f j_normalLimiter_compMass1_linearX_0 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearX[iP_0]);
        V8f j_normalLimiter_compMass1_linearY_0 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearY[iP_0]);
        V8f j_normalLimiter_compMass2_linearX_0 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearX[iP_0]);
        V8f j_normalLimiter_compMass2_linearY_0 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearY[iP_0]);
        V8f j_normalLimiter_compMass1_angular_0 = _mm256_load_ps(&jointP.normalLimiter_compMass1_angular[iP_0]);
        V8f j_normalLimiter_compMass2_angular_0 = _mm256_load_ps(&jointP.normalLimiter_compMass2_angular[iP_0]);
        V8f j_normalLimiter_compInvMass_0 = _mm256_load_ps(&jointP.normalLimiter_compInvMass[iP_0]);
        V8f j_normalLimiter_accumulatedImpulse_0 = _mm256_load_ps(&jointP.normalLimiter_accumulatedImpulse[iP_0]);
        V8f j_normalLimiter_dstVelocity_0 = _mm256_load_ps(&jointP.normalLimiter_dstVelocity[iP_0]);

        V8f j_frictionLimiter_normalProjector1X_0 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector1X[iP_0]);
        V8f j_frictionLimiter_normalProjector1Y_0 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector1Y[iP_0]);
        V8f j_frictionLimiter_normalProjector2X_0 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector2X[iP_0]);
        V8f j_frictionLimiter_normalProjector2Y_0 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector2Y[iP_0]);
        V8f j_frictionLimiter_angularProjector1_0 = _mm256_load_ps(&jointP.frictionLimiter_angularProjector1[iP_0]);
        V8f j_frictionLimiter_angularProjector2_0 = _mm256_load_ps(&jointP.frictionLimiter_angularProjector2[iP_0]);

        V8f j_frictionLimiter_compMass1_linearX_0 = _mm256_load_ps(&jointP.frictionLimiter_compMass1_linearX[iP_0]);
        V8f j_frictionLimiter_compMass1_linearY_0 = _mm256_load_ps(&jointP.frictionLimiter_compMass1_linearY[iP_0]);
        V8f j_frictionLimiter_compMass2_linearX_0 = _mm256_load_ps(&jointP.frictionLimiter_compMass2_linearX[iP_0]);
        V8f j_frictionLimiter_compMass2_linearY_0 = _mm256_load_ps(&jointP.frictionLimiter_compMass2_linearY[iP_0]);
        V8f j_frictionLimiter_compMass1_angular_0 = _mm256_load_ps(&jointP.frictionLimiter_compMass1_angular[iP_0]);
        V8f j_frictionLimiter_compMass2_angular_0 = _mm256_load_ps(&jointP.frictionLimiter_compMass2_angular[iP_0]);
        V8f j_frictionLimiter_compInvMass_0 = _mm256_load_ps(&jointP.frictionLimiter_compInvMass[iP_0]);
        V8f j_frictionLimiter_accumulatedImpulse_0 = _mm256_load_ps(&jointP.frictionLimiter_accumulatedImpulse[iP_0]);

        V8f j_normalLimiter_normalProjector1X_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1X[iP_1]);
        V8f j_normalLimiter_normalProjector1Y_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1Y[iP_1]);
        V8f j_normalLimiter_normalProjector2X_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2X[iP_1]);
        V8f j_normalLimiter_normalProjector2Y_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2Y[iP_1]);
        V8f j_normalLimiter_angularProjector1_1 = _mm256_load_ps(&jointP.normalLimiter_angularProjector1[iP_1]);
        V8f j_normalLimiter_angularProjector2_1 = _mm256_load_ps(&jointP.normalLimiter_angularProjector2[iP_1]);

        V8f j_normalLimiter_compMass1_linearX_1 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearX[iP_1]);
        V8f j_normalLimiter_compMass1_linearY_1 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearY[iP_1]);
        V8f j_normalLimiter_compMass2_linearX_1 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearX[iP_1]);
        V8f j_normalLimiter_compMass2_linearY_1 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearY[iP_1]);
        V8f j_normalLimiter_compMass1_angular_1 = _mm256_load_ps(&jointP.normalLimiter_compMass1_angular[iP_1]);
        V8f j_normalLimiter_compMass2_angular_1 = _mm256_load_ps(&jointP.normalLimiter_compMass2_angular[iP_1]);
        V8f j_normalLimiter_compInvMass_1 = _mm256_load_ps(&jointP.normalLimiter_compInvMass[iP_1]);
        V8f j_normalLimiter_accumulatedImpulse_1 = _mm256_load_ps(&jointP.normalLimiter_accumulatedImpulse[iP_1]);
        V8f j_normalLimiter_dstVelocity_1 = _mm256_load_ps(&jointP.normalLimiter_dstVelocity[iP_1]);

        V8f j_frictionLimiter_normalProjector1X_1 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector1X[iP_1]);
        V8f j_frictionLimiter_normalProjector1Y_1 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector1Y[iP_1]);
        V8f j_frictionLimiter_normalProjector2X_1 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector2X[iP_1]);
        V8f j_frictionLimiter_normalProjector2Y_1 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector2Y[iP_1]);
        V8f j_frictionLimiter_angularProjector1_1 = _mm256_load_ps(&jointP.frictionLimiter_angularProjector1[iP_1]);
        V8f j_frictionLimiter_angularProjector2_1 = _mm256_load_ps(&jointP.frictionLimiter_angularProjector2[iP_1]);

        V8f j_frictionLimiter_compMass1_linearX_1 = _mm256_load_ps(&jointP.frictionLimiter_compMass1_linearX[iP_1]);
        V8f j_frictionLimiter_compMass1_linearY_1 = _mm256_load_ps(&jointP.frictionLimiter_compMass1_linearY[iP_1]);
        V8f j_frictionLimiter_compMass2_linearX_1 = _mm256_load_ps(&jointP.frictionLimiter_compMass2_linearX[iP_1]);
        V8f j_frictionLimiter_compMass2_linearY_1 = _mm256_load_ps(&jointP.frictionLimiter_compMass2_linearY[iP_1]);
        V8f j_frictionLimiter_compMass1_angular_1 = _mm256_load_ps(&jointP.frictionLimiter_compMass1_angular[iP_1]);
        V8f j_frictionLimiter_compMass2_angular_1 = _mm256_load_ps(&jointP.frictionLimiter_compMass2_angular[iP_1]);
        V8f j_frictionLimiter_compInvMass_1 = _mm256_load_ps(&jointP.frictionLimiter_compInvMass[iP_1]);
        V8f j_frictionLimiter_accumulatedImpulse_1 = _mm256_load_ps(&jointP.frictionLimiter_accumulatedImpulse[iP_1]);

        V8f normaldV1_0 = j_normalLimiter_dstVelocity_0;

        normaldV1_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1X_0, body1_velocityX_0, normaldV1_0);
        normaldV1_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1Y_0, body1_velocityY_0, normaldV1_0);
        normaldV1_0 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector1_0, body1_angularVelocity_0, normaldV1_0);

        V8f normaldV2_0 = V8f::zero();

        normaldV2_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2X_0, body2_velocityX_0, normaldV2_0);
        normaldV2_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2Y_0, body2_velocityY_0, normaldV2_0);
        normaldV2_0 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector2_0, body2_angularVelocity_0, normaldV2_0);

        V8f normaldV_0 = _mm256_add_ps(normaldV1_0, normaldV2_0);

        V8f normalDeltaImpulse_0 = _mm256_mul_ps(normaldV_0, j_normalLimiter_compInvMass_0);

        normalDeltaImpulse_0 = _mm256_max_ps(normalDeltaImpulse_0, _mm256_xor_ps(sign, j_normalLimiter_accumulatedImpulse_0));

        body1_velocityX_0 = _mm256_fmadd_ps(j_normalLimiter_compMass1_linearX_0, normalDeltaImpulse_0, body1_velocityX_0);
        body1_velocityY_0 = _mm256_fmadd_ps(j_normalLimiter_compMass1_linearY_0, normalDeltaImpulse_0, body1_velocityY_0);
        body1_angularVelocity_0 = _mm256_fmadd_ps(j_normalLimiter_compMass1_angular_0, normalDeltaImpulse_0, body1_angularVelocity_0);

        body2_velocityX_0 = _mm256_fmadd_ps(j_normalLimiter_compMass2_linearX_0, normalDeltaImpulse_0, body2_velocityX_0);
        body2_velocityY_0 = _mm256_fmadd_ps(j_normalLimiter_compMass2_linearY_0, normalDeltaImpulse_0, body2_velocityY_0);
        body2_angularVelocity_0 = _mm256_fmadd_ps(j_normalLimiter_compMass2_angular_0, normalDeltaImpulse_0, body2_angularVelocity_0);

        j_normalLimiter_accumulatedImpulse_0 = _mm256_add_ps(j_normalLimiter_accumulatedImpulse_0, normalDeltaImpulse_0);

        V8f frictiondV0_0 = V8f::zero();

        frictiondV0_0 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector1X_0, body1_velocityX_0, frictiondV0_0);
        frictiondV0_0 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector1Y_0, body1_velocityY_0, frictiondV0_0);
        frictiondV0_0 = _mm256_fnmadd_ps(j_frictionLimiter_angularProjector1_0, body1_angularVelocity_0, frictiondV0_0);

        V8f frictiondV1_0 = V8f::zero();

        frictiondV1_0 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector2X_0, body2_velocityX_0, frictiondV1_0);
        frictiondV1_0 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector2Y_0, body2_velocityY_0, frictiondV1_0);
        frictiondV1_0 = _mm256_fnmadd_ps(j_frictionLimiter_angularProjector2_0, body2_angularVelocity_0, frictiondV1_0);

        V8f frictiondV_0 = _mm256_add_ps(frictiondV0_0, frictiondV1_0);

        V8f frictionDeltaImpulse_0 = _mm256_mul_ps(frictiondV_0, j_frictionLimiter_compInvMass_0);

        V8f reactionForce_0 = j_normalLimiter_accumulatedImpulse_0;
        V8f accumulatedImpulse_0 = j_frictionLimiter_accumulatedImpulse_0;

        V8f frictionForce_0 = _mm256_add_ps(accumulatedImpulse_0, frictionDeltaImpulse_0);
        V8f reactionForceScaled_0 = _mm256_mul_ps(reactionForce_0, _mm256_set1_ps(kFrictionCoefficient));

        V8f frictionForceAbs_0 = _mm256_andnot_ps(sign, frictionForce_0);
        V8f reactionForceScaledSigned_0 = _mm256_xor_ps(_mm256_and_ps(frictionForce_0, sign), reactionForceScaled_0);
        V8f frictionDeltaImpulseAdjusted_0 = _mm256_sub_ps(reactionForceScaledSigned_0, accumulatedImpulse_0);

        V8f frictionSelector_0 = _mm256_cmp_ps(frictionForceAbs_0, reactionForceScaled_0, _CMP_GT_OQ);

        frictionDeltaImpulse_0 = _mm256_blendv_ps(frictionDeltaImpulse_0, frictionDeltaImpulseAdjusted_0, frictionSelector_0);

        j_frictionLimiter_accumulatedImpulse_0 = _mm256_add_ps(j_frictionLimiter_accumulatedImpulse_0, frictionDeltaImpulse_0);

        body1_velocityX_0 = _mm256_fmadd_ps(j_frictionLimiter_compMass1_linearX_0, frictionDeltaImpulse_0, body1_velocityX_0);
        body1_velocityY_0 = _mm256_fmadd_ps(j_frictionLimiter_compMass1_linearY_0, frictionDeltaImpulse_0, body1_velocityY_0);
        body1_angularVelocity_0 = _mm256_fmadd_ps(j_frictionLimiter_compMass1_angular_0, frictionDeltaImpulse_0, body1_angularVelocity_0);

        body2_velocityX_0 = _mm256_fmadd_ps(j_frictionLimiter_compMass2_linearX_0, frictionDeltaImpulse_0, body2_velocityX_0);
        body2_velocityY_0 = _mm256_fmadd_ps(j_frictionLimiter_compMass2_linearY_0, frictionDeltaImpulse_0, body2_velocityY_0);
        body2_angularVelocity_0 = _mm256_fmadd_ps(j_frictionLimiter_compMass2_angular_0, frictionDeltaImpulse_0, body2_angularVelocity_0);

        V8f normaldV1_1 = j_normalLimiter_dstVelocity_1;

        normaldV1_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1X_1, body1_velocityX_1, normaldV1_1);
        normaldV1_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1Y_1, body1_velocityY_1, normaldV1_1);
        normaldV1_1 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector1_1, body1_angularVelocity_1, normaldV1_1);

        V8f normaldV2_1 = V8f::zero();

        normaldV2_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2X_1, body2_velocityX_1, normaldV2_1);
        normaldV2_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2Y_1, body2_velocityY_1, normaldV2_1);
        normaldV2_1 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector2_1, body2_angularVelocity_1, normaldV2_1);

        V8f normaldV_1 = _mm256_add_ps(normaldV1_1, normaldV2_1);

        V8f normalDeltaImpulse_1 = _mm256_mul_ps(normaldV_1, j_normalLimiter_compInvMass_1);

        normalDeltaImpulse_1 = _mm256_max_ps(normalDeltaImpulse_1, _mm256_xor_ps(sign, j_normalLimiter_accumulatedImpulse_1));

        body1_velocityX_1 = _mm256_fmadd_ps(j_normalLimiter_compMass1_linearX_1, normalDeltaImpulse_1, body1_velocityX_1);
        body1_velocityY_1 = _mm256_fmadd_ps(j_normalLimiter_compMass1_linearY_1, normalDeltaImpulse_1, body1_velocityY_1);
        body1_angularVelocity_1 = _mm256_fmadd_ps(j_normalLimiter_compMass1_angular_1, normalDeltaImpulse_1, body1_angularVelocity_1);

        body2_velocityX_1 = _mm256_fmadd_ps(j_normalLimiter_compMass2_linearX_1, normalDeltaImpulse_1, body2_velocityX_1);
        body2_velocityY_1 = _mm256_fmadd_ps(j_normalLimiter_compMass2_linearY_1, normalDeltaImpulse_1, body2_velocityY_1);
        body2_angularVelocity_1 = _mm256_fmadd_ps(j_normalLimiter_compMass2_angular_1, normalDeltaImpulse_1, body2_angularVelocity_1);

        j_normalLimiter_accumulatedImpulse_1 = _mm256_add_ps(j_normalLimiter_accumulatedImpulse_1, normalDeltaImpulse_1);

        V8f frictiondV0_1 = V8f::zero();

        frictiondV0_1 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector1X_1, body1_velocityX_1, frictiondV0_1);
        frictiondV0_1 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector1Y_1, body1_velocityY_1, frictiondV0_1);
        frictiondV0_1 = _mm256_fnmadd_ps(j_frictionLimiter_angularProjector1_1, body1_angularVelocity_1, frictiondV0_1);

        V8f frictiondV1_1 = V8f::zero();

        frictiondV1_1 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector2X_1, body2_velocityX_1, frictiondV1_1);
        frictiondV1_1 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector2Y_1, body2_velocityY_1, frictiondV1_1);
        frictiondV1_1 = _mm256_fnmadd_ps(j_frictionLimiter_angularProjector2_1, body2_angularVelocity_1, frictiondV1_1);

        V8f frictiondV_1 = _mm256_add_ps(frictiondV0_1, frictiondV1_1);

        V8f frictionDeltaImpulse_1 = _mm256_mul_ps(frictiondV_1, j_frictionLimiter_compInvMass_1);

        V8f reactionForce_1 = j_normalLimiter_accumulatedImpulse_1;
        V8f accumulatedImpulse_1 = j_frictionLimiter_accumulatedImpulse_1;

        V8f frictionForce_1 = _mm256_add_ps(accumulatedImpulse_1, frictionDeltaImpulse_1);
        V8f reactionForceScaled_1 = _mm256_mul_ps(reactionForce_1, _mm256_set1_ps(kFrictionCoefficient));

        V8f frictionForceAbs_1 = _mm256_andnot_ps(sign, frictionForce_1);
        V8f reactionForceScaledSigned_1 = _mm256_xor_ps(_mm256_and_ps(frictionForce_1, sign), reactionForceScaled_1);
        V8f frictionDeltaImpulseAdjusted_1 = _mm256_sub_ps(reactionForceScaledSigned_1, accumulatedImpulse_1);

        V8f frictionSelector_1 = _mm256_cmp_ps(frictionForceAbs_1, reactionForceScaled_1, _CMP_GT_OQ);

        frictionDeltaImpulse_1 = _mm256_blendv_ps(frictionDeltaImpulse_1, frictionDeltaImpulseAdjusted_1, frictionSelector_1);

        j_frictionLimiter_accumulatedImpulse_1 = _mm256_add_ps(j_frictionLimiter_accumulatedImpulse_1, frictionDeltaImpulse_1);

        body1_velocityX_1 = _mm256_fmadd_ps(j_frictionLimiter_compMass1_linearX_1, frictionDeltaImpulse_1, body1_velocityX_1);
        body1_velocityY_1 = _mm256_fmadd_ps(j_frictionLimiter_compMass1_linearY_1, frictionDeltaImpulse_1, body1_velocityY_1);
        body1_angularVelocity_1 = _mm256_fmadd_ps(j_frictionLimiter_compMass1_angular_1, frictionDeltaImpulse_1, body1_angularVelocity_1);

        body2_velocityX_1 = _mm256_fmadd_ps(j_frictionLimiter_compMass2_linearX_1, frictionDeltaImpulse_1, body2_velocityX_1);
        body2_velocityY_1 = _mm256_fmadd_ps(j_frictionLimiter_compMass2_linearY_1, frictionDeltaImpulse_1, body2_velocityY_1);
        body2_angularVelocity_1 = _mm256_fmadd_ps(j_frictionLimiter_compMass2_angular_1, frictionDeltaImpulse_1, body2_angularVelocity_1);

        _mm256_store_ps(&jointP.normalLimiter_accumulatedImpulse[iP_0], j_normalLimiter_accumulatedImpulse_0);
        _mm256_store_ps(&jointP.frictionLimiter_accumulatedImpulse[iP_0], j_frictionLimiter_accumulatedImpulse_0);

        _mm256_store_ps(&jointP.normalLimiter_accumulatedImpulse[iP_1], j_normalLimiter_accumulatedImpulse_1);
        _mm256_store_ps(&jointP.frictionLimiter_accumulatedImpulse[iP_1], j_frictionLimiter_accumulatedImpulse_1);

        V8f cumulativeImpulse_0 = _mm256_max_ps(_mm256_andnot_ps(sign, normalDeltaImpulse_0), _mm256_andnot_ps(sign, frictionDeltaImpulse_0));
        V8f cumulativeImpulse_1 = _mm256_max_ps(_mm256_andnot_ps(sign, normalDeltaImpulse_1), _mm256_andnot_ps(sign, frictionDeltaImpulse_1));

        V8f productive_0 = _mm256_cmp_ps(cumulativeImpulse_0, _mm256_set1_ps(kProductiveImpulse), _CMP_GT_OQ);
        V8f productive_1 = _mm256_cmp_ps(cumulativeImpulse_1, _mm256_set1_ps(kProductiveImpulse), _CMP_GT_OQ);

        productive_any = _mm256_or_si256(productive_any, _mm256_or_si256(productive_0, productive_1));

        body1_lastIteration_0 = _mm256_blendv_epi8(body1_lastIteration_0, iterationIndex0, productive_0);
        body2_lastIteration_0 = _mm256_blendv_epi8(body2_lastIteration_0, iterationIndex0, productive_0);

        body1_lastIteration_1 = _mm256_blendv_epi8(body1_lastIteration_1, iterationIndex0, productive_1);
        body2_lastIteration_1 = _mm256_blendv_epi8(body2_lastIteration_1, iterationIndex0, productive_1);

        // this is a bit painful :(
        static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Storing assumes fixed layout");

        row0 = body1_velocityX_0;
        row1 = body1_velocityY_0;
        row2 = body1_angularVelocity_0;
        row3 = bitcast(body1_lastIteration_0);

        row4 = body2_velocityX_0;
        row5 = body2_velocityY_0;
        row6 = body2_angularVelocity_0;
        row7 = bitcast(body2_lastIteration_0);

        _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_0 + 0]].velocity.x, _mm256_extractf128_ps(row0, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_0 + 0]].velocity.x, _mm256_extractf128_ps(row0, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_0 + 1]].velocity.x, _mm256_extractf128_ps(row1, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_0 + 1]].velocity.x, _mm256_extractf128_ps(row1, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_0 + 2]].velocity.x, _mm256_extractf128_ps(row2, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_0 + 2]].velocity.x, _mm256_extractf128_ps(row2, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_0 + 3]].velocity.x, _mm256_extractf128_ps(row3, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_0 + 3]].velocity.x, _mm256_extractf128_ps(row3, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_0 + 4]].velocity.x, _mm256_extractf128_ps(row4, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_0 + 4]].velocity.x, _mm256_extractf128_ps(row4, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_0 + 5]].velocity.x, _mm256_extractf128_ps(row5, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_0 + 5]].velocity.x, _mm256_extractf128_ps(row5, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_0 + 6]].velocity.x, _mm256_extractf128_ps(row6, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_0 + 6]].velocity.x, _mm256_extractf128_ps(row6, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_0 + 7]].velocity.x, _mm256_extractf128_ps(row7, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_0 + 7]].velocity.x, _mm256_extractf128_ps(row7, 1));

        row0 = body1_velocityX_1;
        row1 = body1_velocityY_1;
        row2 = body1_angularVelocity_1;
        row3 = bitcast(body1_lastIteration_1);

        row4 = body2_velocityX_1;
        row5 = body2_velocityY_1;
        row6 = body2_angularVelocity_1;
        row7 = bitcast(body2_lastIteration_1);

        _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_1 + 0]].velocity.x, _mm256_extractf128_ps(row0, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_1 + 0]].velocity.x, _mm256_extractf128_ps(row0, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_1 + 1]].velocity.x, _mm256_extractf128_ps(row1, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_1 + 1]].velocity.x, _mm256_extractf128_ps(row1, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_1 + 2]].velocity.x, _mm256_extractf128_ps(row2, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_1 + 2]].velocity.x, _mm256_extractf128_ps(row2, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_1 + 3]].velocity.x, _mm256_extractf128_ps(row3, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_1 + 3]].velocity.x, _mm256_extractf128_ps(row3, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_1 + 4]].velocity.x, _mm256_extractf128_ps(row4, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_1 + 4]].velocity.x, _mm256_extractf128_ps(row4, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_1 + 5]].velocity.x, _mm256_extractf128_ps(row5, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_1 + 5]].velocity.x, _mm256_extractf128_ps(row5, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_1 + 6]].velocity.x, _mm256_extractf128_ps(row6, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_1 + 6]].velocity.x, _mm256_extractf128_ps(row6, 1));

        _mm_store_ps(&solveBodiesImpulse[jointP.body1Index[iP_1 + 7]].velocity.x, _mm256_extractf128_ps(row7, 0));
        _mm_store_ps(&solveBodiesImpulse[jointP.body2Index[iP_1 + 7]].velocity.x, _mm256_extractf128_ps(row7, 1));
    }

    return _mm256_movemask_epi8(productive_any) != 0;
}

#endif

template <int N>
NOINLINE bool Solver::SolveJointsDisplacementSoA(ContactJointPacked<N>* joint_packed, int jointStart, int jointCount, int iterationIndex)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsDisplacementSoA", -1);

    bool productive_any = false;

    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex++)
    {
        int i = jointIndex;

        ContactJointPacked<N>& jointP = joint_packed[unsigned(i) / N];
        int iP = i & (N - 1);

        SolveBody* body1 = &solveBodiesDisplacement[jointP.body1Index[iP]];
        SolveBody* body2 = &solveBodiesDisplacement[jointP.body2Index[iP]];

        if (body1->lastIteration < iterationIndex - 1 && body2->lastIteration < iterationIndex - 1)
            continue;

        float dV = jointP.normalLimiter_dstDisplacingVelocity[iP];

        dV -= jointP.normalLimiter_normalProjector1X[iP] * body1->velocity.x;
        dV -= jointP.normalLimiter_normalProjector1Y[iP] * body1->velocity.y;
        dV -= jointP.normalLimiter_angularProjector1[iP] * body1->angularVelocity;

        dV -= jointP.normalLimiter_normalProjector2X[iP] * body2->velocity.x;
        dV -= jointP.normalLimiter_normalProjector2Y[iP] * body2->velocity.y;
        dV -= jointP.normalLimiter_angularProjector2[iP] * body2->angularVelocity;

        float displacingDeltaImpulse = dV * jointP.normalLimiter_compInvMass[iP];

        if (displacingDeltaImpulse + jointP.normalLimiter_accumulatedDisplacingImpulse[iP] < 0.0f)
            displacingDeltaImpulse = -jointP.normalLimiter_accumulatedDisplacingImpulse[iP];

        body1->velocity.x += jointP.normalLimiter_compMass1_linearX[iP] * displacingDeltaImpulse;
        body1->velocity.y += jointP.normalLimiter_compMass1_linearY[iP] * displacingDeltaImpulse;
        body1->angularVelocity += jointP.normalLimiter_compMass1_angular[iP] * displacingDeltaImpulse;

        body2->velocity.x += jointP.normalLimiter_compMass2_linearX[iP] * displacingDeltaImpulse;
        body2->velocity.y += jointP.normalLimiter_compMass2_linearY[iP] * displacingDeltaImpulse;
        body2->angularVelocity += jointP.normalLimiter_compMass2_angular[iP] * displacingDeltaImpulse;

        jointP.normalLimiter_accumulatedDisplacingImpulse[iP] += displacingDeltaImpulse;

        if (fabsf(displacingDeltaImpulse) > kProductiveImpulse)
        {
            body1->lastIteration = iterationIndex;
            body2->lastIteration = iterationIndex;
            productive_any = true;
        }
    }

    return productive_any;
}

NOINLINE bool Solver::SolveJointsDisplacementSoA_SSE2(ContactJointPacked<4>* joint_packed, int jointStart, int jointCount, int iterationIndex)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsDisplacementSoA_SSE2", -1);

    assert(jointStart % 4 == 0 && jointCount % 4 == 0);

    V4i iterationIndex0 = V4i::one(iterationIndex);
    V4i iterationIndex2 = V4i::one(iterationIndex - 2);

    V4b productive_any = V4b::zero();

    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 4)
    {
        int i = jointIndex;

        ContactJointPacked<4>& jointP = joint_packed[i >> 2];
        int iP = 0;

        V4f row0, row1, row2, row3;

        static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Loading assumes fixed layout");

        row0 = _mm_load_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 0]].velocity.x);
        row1 = _mm_load_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 1]].velocity.x);
        row2 = _mm_load_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 2]].velocity.x);
        row3 = _mm_load_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 3]].velocity.x);

        _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

        V4f body1_displacingVelocityX = row0;
        V4f body1_displacingVelocityY = row1;
        V4f body1_displacingAngularVelocity = row2;
        V4i body1_lastDisplacementIteration = bitcast(row3);

        row0 = _mm_load_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 0]].velocity.x);
        row1 = _mm_load_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 1]].velocity.x);
        row2 = _mm_load_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 2]].velocity.x);
        row3 = _mm_load_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 3]].velocity.x);

        _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

        V4f body2_displacingVelocityX = row0;
        V4f body2_displacingVelocityY = row1;
        V4f body2_displacingAngularVelocity = row2;
        V4i body2_lastDisplacementIteration = bitcast(row3);

        V4b body1_productive = body1_lastDisplacementIteration > iterationIndex2;
        V4b body2_productive = body2_lastDisplacementIteration > iterationIndex2;
        V4b body_productive = body1_productive | body2_productive;

        if (none(body_productive))
            continue;

        V4f j_normalLimiter_normalProjector1X = _mm_load_ps(&jointP.normalLimiter_normalProjector1X[iP]);
        V4f j_normalLimiter_normalProjector1Y = _mm_load_ps(&jointP.normalLimiter_normalProjector1Y[iP]);
        V4f j_normalLimiter_normalProjector2X = _mm_load_ps(&jointP.normalLimiter_normalProjector2X[iP]);
        V4f j_normalLimiter_normalProjector2Y = _mm_load_ps(&jointP.normalLimiter_normalProjector2Y[iP]);
        V4f j_normalLimiter_angularProjector1 = _mm_load_ps(&jointP.normalLimiter_angularProjector1[iP]);
        V4f j_normalLimiter_angularProjector2 = _mm_load_ps(&jointP.normalLimiter_angularProjector2[iP]);

        V4f j_normalLimiter_compMass1_linearX = _mm_load_ps(&jointP.normalLimiter_compMass1_linearX[iP]);
        V4f j_normalLimiter_compMass1_linearY = _mm_load_ps(&jointP.normalLimiter_compMass1_linearY[iP]);
        V4f j_normalLimiter_compMass2_linearX = _mm_load_ps(&jointP.normalLimiter_compMass2_linearX[iP]);
        V4f j_normalLimiter_compMass2_linearY = _mm_load_ps(&jointP.normalLimiter_compMass2_linearY[iP]);
        V4f j_normalLimiter_compMass1_angular = _mm_load_ps(&jointP.normalLimiter_compMass1_angular[iP]);
        V4f j_normalLimiter_compMass2_angular = _mm_load_ps(&jointP.normalLimiter_compMass2_angular[iP]);
        V4f j_normalLimiter_compInvMass = _mm_load_ps(&jointP.normalLimiter_compInvMass[iP]);
        V4f j_normalLimiter_dstDisplacingVelocity = _mm_load_ps(&jointP.normalLimiter_dstDisplacingVelocity[iP]);
        V4f j_normalLimiter_accumulatedDisplacingImpulse = _mm_load_ps(&jointP.normalLimiter_accumulatedDisplacingImpulse[iP]);

        V4f dV = j_normalLimiter_dstDisplacingVelocity;

        dV -= j_normalLimiter_normalProjector1X * body1_displacingVelocityX;
        dV -= j_normalLimiter_normalProjector1Y * body1_displacingVelocityY;
        dV -= j_normalLimiter_angularProjector1 * body1_displacingAngularVelocity;

        dV -= j_normalLimiter_normalProjector2X * body2_displacingVelocityX;
        dV -= j_normalLimiter_normalProjector2Y * body2_displacingVelocityY;
        dV -= j_normalLimiter_angularProjector2 * body2_displacingAngularVelocity;

        V4f displacingDeltaImpulse = dV * j_normalLimiter_compInvMass;

        displacingDeltaImpulse = max(displacingDeltaImpulse, -j_normalLimiter_accumulatedDisplacingImpulse);

        body1_displacingVelocityX += j_normalLimiter_compMass1_linearX * displacingDeltaImpulse;
        body1_displacingVelocityY += j_normalLimiter_compMass1_linearY * displacingDeltaImpulse;
        body1_displacingAngularVelocity += j_normalLimiter_compMass1_angular * displacingDeltaImpulse;

        body2_displacingVelocityX += j_normalLimiter_compMass2_linearX * displacingDeltaImpulse;
        body2_displacingVelocityY += j_normalLimiter_compMass2_linearY * displacingDeltaImpulse;
        body2_displacingAngularVelocity += j_normalLimiter_compMass2_angular * displacingDeltaImpulse;

        j_normalLimiter_accumulatedDisplacingImpulse += displacingDeltaImpulse;

        _mm_store_ps(&jointP.normalLimiter_accumulatedDisplacingImpulse[iP], j_normalLimiter_accumulatedDisplacingImpulse);

        V4b productive = abs(displacingDeltaImpulse) > V4f::one(kProductiveImpulse);

        productive_any |= productive;

        body1_lastDisplacementIteration = select(body1_lastDisplacementIteration, iterationIndex0, productive);
        body2_lastDisplacementIteration = select(body2_lastDisplacementIteration, iterationIndex0, productive);

        // this is a bit painful :(
        static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Storing assumes fixed layout");

        row0 = body1_displacingVelocityX;
        row1 = body1_displacingVelocityY;
        row2 = body1_displacingAngularVelocity;
        row3 = bitcast(body1_lastDisplacementIteration);

        _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 0]].velocity.x, row0);
        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 1]].velocity.x, row1);
        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 2]].velocity.x, row2);
        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 3]].velocity.x, row3);

        row0 = body2_displacingVelocityX;
        row1 = body2_displacingVelocityY;
        row2 = body2_displacingAngularVelocity;
        row3 = bitcast(body2_lastDisplacementIteration);

        _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 0]].velocity.x, row0);
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 1]].velocity.x, row1);
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 2]].velocity.x, row2);
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 3]].velocity.x, row3);
    }

    return any(productive_any);
}

#ifdef __AVX2__
NOINLINE bool Solver::SolveJointsDisplacementSoA_AVX2(ContactJointPacked<8>* joint_packed, int jointStart, int jointCount, int iterationIndex)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsDisplacementSoA_AVX2", -1);

    assert(jointStart % 8 == 0 && jointCount % 8 == 0);

    V8f sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    V8i iterationIndex0 = _mm256_set1_epi32(iterationIndex);
    V8i iterationIndex2 = _mm256_set1_epi32(iterationIndex - 2);

    V8i productive_any = _mm256_setzero_si256();

    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 8)
    {
        int i = jointIndex;

        ContactJointPacked<8>& jointP = joint_packed[i >> 3];
        int iP = 0;

        V8f row0, row1, row2, row3, row4, row5, row6, row7;

        static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Loading assumes fixed layout");

        row0 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP + 0]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP + 0]].velocity.x);
        row1 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP + 1]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP + 1]].velocity.x);
        row2 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP + 2]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP + 2]].velocity.x);
        row3 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP + 3]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP + 3]].velocity.x);
        row4 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP + 4]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP + 4]].velocity.x);
        row5 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP + 5]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP + 5]].velocity.x);
        row6 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP + 6]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP + 6]].velocity.x);
        row7 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP + 7]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP + 7]].velocity.x);

        _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

        V8f body1_displacingVelocityX = row0;
        V8f body1_displacingVelocityY = row1;
        V8f body1_displacingAngularVelocity = row2;
        V8i body1_lastDisplacementIteration = bitcast(row3);

        V8f body2_displacingVelocityX = row4;
        V8f body2_displacingVelocityY = row5;
        V8f body2_displacingAngularVelocity = row6;
        V8i body2_lastDisplacementIteration = bitcast(row7);

        V8i body_lastDisplacementIteration = _mm256_max_epi32(body1_lastDisplacementIteration, body2_lastDisplacementIteration);
        V8i body_productive = _mm256_cmpgt_epi32(body_lastDisplacementIteration, iterationIndex2);

        if (_mm256_movemask_epi8(body_productive) == 0)
            continue;

        V8f j_normalLimiter_normalProjector1X = _mm256_load_ps(&jointP.normalLimiter_normalProjector1X[iP]);
        V8f j_normalLimiter_normalProjector1Y = _mm256_load_ps(&jointP.normalLimiter_normalProjector1Y[iP]);
        V8f j_normalLimiter_normalProjector2X = _mm256_load_ps(&jointP.normalLimiter_normalProjector2X[iP]);
        V8f j_normalLimiter_normalProjector2Y = _mm256_load_ps(&jointP.normalLimiter_normalProjector2Y[iP]);
        V8f j_normalLimiter_angularProjector1 = _mm256_load_ps(&jointP.normalLimiter_angularProjector1[iP]);
        V8f j_normalLimiter_angularProjector2 = _mm256_load_ps(&jointP.normalLimiter_angularProjector2[iP]);

        V8f j_normalLimiter_compMass1_linearX = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearX[iP]);
        V8f j_normalLimiter_compMass1_linearY = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearY[iP]);
        V8f j_normalLimiter_compMass2_linearX = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearX[iP]);
        V8f j_normalLimiter_compMass2_linearY = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearY[iP]);
        V8f j_normalLimiter_compMass1_angular = _mm256_load_ps(&jointP.normalLimiter_compMass1_angular[iP]);
        V8f j_normalLimiter_compMass2_angular = _mm256_load_ps(&jointP.normalLimiter_compMass2_angular[iP]);
        V8f j_normalLimiter_compInvMass = _mm256_load_ps(&jointP.normalLimiter_compInvMass[iP]);
        V8f j_normalLimiter_dstDisplacingVelocity = _mm256_load_ps(&jointP.normalLimiter_dstDisplacingVelocity[iP]);
        V8f j_normalLimiter_accumulatedDisplacingImpulse = _mm256_load_ps(&jointP.normalLimiter_accumulatedDisplacingImpulse[iP]);

        V8f dV = j_normalLimiter_dstDisplacingVelocity;

        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector1X, body1_displacingVelocityX));
        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector1Y, body1_displacingVelocityY));
        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_angularProjector1, body1_displacingAngularVelocity));

        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector2X, body2_displacingVelocityX));
        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector2Y, body2_displacingVelocityY));
        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_angularProjector2, body2_displacingAngularVelocity));

        V8f displacingDeltaImpulse = _mm256_mul_ps(dV, j_normalLimiter_compInvMass);

        displacingDeltaImpulse = _mm256_max_ps(displacingDeltaImpulse, _mm256_xor_ps(sign, j_normalLimiter_accumulatedDisplacingImpulse));

        body1_displacingVelocityX = _mm256_add_ps(body1_displacingVelocityX, _mm256_mul_ps(j_normalLimiter_compMass1_linearX, displacingDeltaImpulse));
        body1_displacingVelocityY = _mm256_add_ps(body1_displacingVelocityY, _mm256_mul_ps(j_normalLimiter_compMass1_linearY, displacingDeltaImpulse));
        body1_displacingAngularVelocity = _mm256_add_ps(body1_displacingAngularVelocity, _mm256_mul_ps(j_normalLimiter_compMass1_angular, displacingDeltaImpulse));

        body2_displacingVelocityX = _mm256_add_ps(body2_displacingVelocityX, _mm256_mul_ps(j_normalLimiter_compMass2_linearX, displacingDeltaImpulse));
        body2_displacingVelocityY = _mm256_add_ps(body2_displacingVelocityY, _mm256_mul_ps(j_normalLimiter_compMass2_linearY, displacingDeltaImpulse));
        body2_displacingAngularVelocity = _mm256_add_ps(body2_displacingAngularVelocity, _mm256_mul_ps(j_normalLimiter_compMass2_angular, displacingDeltaImpulse));

        j_normalLimiter_accumulatedDisplacingImpulse = _mm256_add_ps(j_normalLimiter_accumulatedDisplacingImpulse, displacingDeltaImpulse);

        _mm256_store_ps(&jointP.normalLimiter_accumulatedDisplacingImpulse[iP], j_normalLimiter_accumulatedDisplacingImpulse);

        V8i productive = bitcast(V8f(_mm256_cmp_ps(_mm256_andnot_ps(sign, displacingDeltaImpulse), _mm256_set1_ps(kProductiveImpulse), _CMP_GT_OQ)));

        productive_any = _mm256_or_si256(productive_any, productive);

        body1_lastDisplacementIteration = _mm256_blendv_epi8(body1_lastDisplacementIteration, iterationIndex0, productive);
        body2_lastDisplacementIteration = _mm256_blendv_epi8(body2_lastDisplacementIteration, iterationIndex0, productive);

        // this is a bit painful :(
        static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Storing assumes fixed layout");

        row0 = body1_displacingVelocityX;
        row1 = body1_displacingVelocityY;
        row2 = body1_displacingAngularVelocity;
        row3 = bitcast(body1_lastDisplacementIteration);

        row4 = body2_displacingVelocityX;
        row5 = body2_displacingVelocityY;
        row6 = body2_displacingAngularVelocity;
        row7 = bitcast(body2_lastDisplacementIteration);

        _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 0]].velocity.x, _mm256_extractf128_ps(row0, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 0]].velocity.x, _mm256_extractf128_ps(row0, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 1]].velocity.x, _mm256_extractf128_ps(row1, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 1]].velocity.x, _mm256_extractf128_ps(row1, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 2]].velocity.x, _mm256_extractf128_ps(row2, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 2]].velocity.x, _mm256_extractf128_ps(row2, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 3]].velocity.x, _mm256_extractf128_ps(row3, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 3]].velocity.x, _mm256_extractf128_ps(row3, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 4]].velocity.x, _mm256_extractf128_ps(row4, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 4]].velocity.x, _mm256_extractf128_ps(row4, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 5]].velocity.x, _mm256_extractf128_ps(row5, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 5]].velocity.x, _mm256_extractf128_ps(row5, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 6]].velocity.x, _mm256_extractf128_ps(row6, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 6]].velocity.x, _mm256_extractf128_ps(row6, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP + 7]].velocity.x, _mm256_extractf128_ps(row7, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP + 7]].velocity.x, _mm256_extractf128_ps(row7, 1));
    }

    return _mm256_movemask_epi8(productive_any) != 0;
}
#endif

#if defined(__AVX2__) && defined(__FMA__)
NOINLINE bool Solver::SolveJointsDisplacementSoA_FMA(ContactJointPacked<16>* joint_packed, int jointStart, int jointCount, int iterationIndex)
{
    MICROPROFILE_SCOPEI("Physics", "SolveJointsDisplacementSoA_FMA", -1);

    assert(jointStart % 16 == 0 && jointCount % 16 == 0);

    V8f sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    V8i iterationIndex0 = _mm256_set1_epi32(iterationIndex);
    V8i iterationIndex2 = _mm256_set1_epi32(iterationIndex - 2);

    V8i productive_any = _mm256_setzero_si256();

    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 16)
    {
        int i = jointIndex;

        ContactJointPacked<16>& jointP = joint_packed[i >> 4];
        int iP_0 = 0;
        int iP_1 = 8;

        V8f row0, row1, row2, row3, row4, row5, row6, row7;

        static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Loading assumes fixed layout");

        row0 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 0]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_0 + 0]].velocity.x);
        row1 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 1]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_0 + 1]].velocity.x);
        row2 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 2]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_0 + 2]].velocity.x);
        row3 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 3]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_0 + 3]].velocity.x);
        row4 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 4]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_0 + 4]].velocity.x);
        row5 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 5]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_0 + 5]].velocity.x);
        row6 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 6]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_0 + 6]].velocity.x);
        row7 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 7]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_0 + 7]].velocity.x);

        _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

        V8f body1_displacingVelocityX_0 = row0;
        V8f body1_displacingVelocityY_0 = row1;
        V8f body1_displacingAngularVelocity_0 = row2;
        V8i body1_lastDisplacementIteration_0 = bitcast(row3);

        V8f body2_displacingVelocityX_0 = row4;
        V8f body2_displacingVelocityY_0 = row5;
        V8f body2_displacingAngularVelocity_0 = row6;
        V8i body2_lastDisplacementIteration_0 = bitcast(row7);

        row0 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 0]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_1 + 0]].velocity.x);
        row1 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 1]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_1 + 1]].velocity.x);
        row2 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 2]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_1 + 2]].velocity.x);
        row3 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 3]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_1 + 3]].velocity.x);
        row4 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 4]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_1 + 4]].velocity.x);
        row5 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 5]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_1 + 5]].velocity.x);
        row6 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 6]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_1 + 6]].velocity.x);
        row7 = _mm256_load2_m128(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 7]].velocity.x, &solveBodiesDisplacement[jointP.body2Index[iP_1 + 7]].velocity.x);

        _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

        V8f body1_displacingVelocityX_1 = row0;
        V8f body1_displacingVelocityY_1 = row1;
        V8f body1_displacingAngularVelocity_1 = row2;
        V8i body1_lastDisplacementIteration_1 = bitcast(row3);

        V8f body2_displacingVelocityX_1 = row4;
        V8f body2_displacingVelocityY_1 = row5;
        V8f body2_displacingAngularVelocity_1 = row6;
        V8i body2_lastDisplacementIteration_1 = bitcast(row7);

        V8i body_lastDisplacementIteration_0 = _mm256_max_epi32(body1_lastDisplacementIteration_0, body2_lastDisplacementIteration_0);
        V8i body_lastDisplacementIteration_1 = _mm256_max_epi32(body1_lastDisplacementIteration_1, body2_lastDisplacementIteration_1);

        V8i body_productive_0 = _mm256_cmpgt_epi32(body_lastDisplacementIteration_0, iterationIndex2);
        V8i body_productive_1 = _mm256_cmpgt_epi32(body_lastDisplacementIteration_1, iterationIndex2);
        V8i body_productive = _mm256_or_si256(body_productive_0, body_productive_1);

        if (_mm256_movemask_epi8(body_productive) == 0)
            continue;

        V8f j_normalLimiter_normalProjector1X_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1X[iP_0]);
        V8f j_normalLimiter_normalProjector1Y_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1Y[iP_0]);
        V8f j_normalLimiter_normalProjector2X_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2X[iP_0]);
        V8f j_normalLimiter_normalProjector2Y_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2Y[iP_0]);
        V8f j_normalLimiter_angularProjector1_0 = _mm256_load_ps(&jointP.normalLimiter_angularProjector1[iP_0]);
        V8f j_normalLimiter_angularProjector2_0 = _mm256_load_ps(&jointP.normalLimiter_angularProjector2[iP_0]);

        V8f j_normalLimiter_compMass1_linearX_0 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearX[iP_0]);
        V8f j_normalLimiter_compMass1_linearY_0 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearY[iP_0]);
        V8f j_normalLimiter_compMass2_linearX_0 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearX[iP_0]);
        V8f j_normalLimiter_compMass2_linearY_0 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearY[iP_0]);
        V8f j_normalLimiter_compMass1_angular_0 = _mm256_load_ps(&jointP.normalLimiter_compMass1_angular[iP_0]);
        V8f j_normalLimiter_compMass2_angular_0 = _mm256_load_ps(&jointP.normalLimiter_compMass2_angular[iP_0]);
        V8f j_normalLimiter_compInvMass_0 = _mm256_load_ps(&jointP.normalLimiter_compInvMass[iP_0]);
        V8f j_normalLimiter_dstDisplacingVelocity_0 = _mm256_load_ps(&jointP.normalLimiter_dstDisplacingVelocity[iP_0]);
        V8f j_normalLimiter_accumulatedDisplacingImpulse_0 = _mm256_load_ps(&jointP.normalLimiter_accumulatedDisplacingImpulse[iP_0]);

        V8f j_normalLimiter_normalProjector1X_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1X[iP_1]);
        V8f j_normalLimiter_normalProjector1Y_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1Y[iP_1]);
        V8f j_normalLimiter_normalProjector2X_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2X[iP_1]);
        V8f j_normalLimiter_normalProjector2Y_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2Y[iP_1]);
        V8f j_normalLimiter_angularProjector1_1 = _mm256_load_ps(&jointP.normalLimiter_angularProjector1[iP_1]);
        V8f j_normalLimiter_angularProjector2_1 = _mm256_load_ps(&jointP.normalLimiter_angularProjector2[iP_1]);

        V8f j_normalLimiter_compMass1_linearX_1 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearX[iP_1]);
        V8f j_normalLimiter_compMass1_linearY_1 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearY[iP_1]);
        V8f j_normalLimiter_compMass2_linearX_1 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearX[iP_1]);
        V8f j_normalLimiter_compMass2_linearY_1 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearY[iP_1]);
        V8f j_normalLimiter_compMass1_angular_1 = _mm256_load_ps(&jointP.normalLimiter_compMass1_angular[iP_1]);
        V8f j_normalLimiter_compMass2_angular_1 = _mm256_load_ps(&jointP.normalLimiter_compMass2_angular[iP_1]);
        V8f j_normalLimiter_compInvMass_1 = _mm256_load_ps(&jointP.normalLimiter_compInvMass[iP_1]);
        V8f j_normalLimiter_dstDisplacingVelocity_1 = _mm256_load_ps(&jointP.normalLimiter_dstDisplacingVelocity[iP_1]);
        V8f j_normalLimiter_accumulatedDisplacingImpulse_1 = _mm256_load_ps(&jointP.normalLimiter_accumulatedDisplacingImpulse[iP_1]);

        V8f dV0_0 = j_normalLimiter_dstDisplacingVelocity_0;

        dV0_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1X_0, body1_displacingVelocityX_0, dV0_0);
        dV0_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1Y_0, body1_displacingVelocityY_0, dV0_0);
        dV0_0 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector1_0, body1_displacingAngularVelocity_0, dV0_0);

        V8f dV1_0 = V8f::zero();

        dV1_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2X_0, body2_displacingVelocityX_0, dV1_0);
        dV1_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2Y_0, body2_displacingVelocityY_0, dV1_0);
        dV1_0 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector2_0, body2_displacingAngularVelocity_0, dV1_0);

        V8f dV_0 = _mm256_add_ps(dV0_0, dV1_0);

        V8f displacingDeltaImpulse_0 = _mm256_mul_ps(dV_0, j_normalLimiter_compInvMass_0);

        displacingDeltaImpulse_0 = _mm256_max_ps(displacingDeltaImpulse_0, _mm256_xor_ps(sign, j_normalLimiter_accumulatedDisplacingImpulse_0));

        body1_displacingVelocityX_0 = _mm256_fmadd_ps(j_normalLimiter_compMass1_linearX_0, displacingDeltaImpulse_0, body1_displacingVelocityX_0);
        body1_displacingVelocityY_0 = _mm256_fmadd_ps(j_normalLimiter_compMass1_linearY_0, displacingDeltaImpulse_0, body1_displacingVelocityY_0);
        body1_displacingAngularVelocity_0 = _mm256_fmadd_ps(j_normalLimiter_compMass1_angular_0, displacingDeltaImpulse_0, body1_displacingAngularVelocity_0);

        body2_displacingVelocityX_0 = _mm256_fmadd_ps(j_normalLimiter_compMass2_linearX_0, displacingDeltaImpulse_0, body2_displacingVelocityX_0);
        body2_displacingVelocityY_0 = _mm256_fmadd_ps(j_normalLimiter_compMass2_linearY_0, displacingDeltaImpulse_0, body2_displacingVelocityY_0);
        body2_displacingAngularVelocity_0 = _mm256_fmadd_ps(j_normalLimiter_compMass2_angular_0, displacingDeltaImpulse_0, body2_displacingAngularVelocity_0);

        j_normalLimiter_accumulatedDisplacingImpulse_0 = _mm256_add_ps(j_normalLimiter_accumulatedDisplacingImpulse_0, displacingDeltaImpulse_0);

        V8f dV0_1 = j_normalLimiter_dstDisplacingVelocity_1;

        dV0_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1X_1, body1_displacingVelocityX_1, dV0_1);
        dV0_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1Y_1, body1_displacingVelocityY_1, dV0_1);
        dV0_1 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector1_1, body1_displacingAngularVelocity_1, dV0_1);

        V8f dV1_1 = V8f::zero();

        dV1_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2X_1, body2_displacingVelocityX_1, dV1_1);
        dV1_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2Y_1, body2_displacingVelocityY_1, dV1_1);
        dV1_1 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector2_1, body2_displacingAngularVelocity_1, dV1_1);

        V8f dV_1 = _mm256_add_ps(dV0_1, dV1_1);

        V8f displacingDeltaImpulse_1 = _mm256_mul_ps(dV_1, j_normalLimiter_compInvMass_1);

        displacingDeltaImpulse_1 = _mm256_max_ps(displacingDeltaImpulse_1, _mm256_xor_ps(sign, j_normalLimiter_accumulatedDisplacingImpulse_1));

        body1_displacingVelocityX_1 = _mm256_fmadd_ps(j_normalLimiter_compMass1_linearX_1, displacingDeltaImpulse_1, body1_displacingVelocityX_1);
        body1_displacingVelocityY_1 = _mm256_fmadd_ps(j_normalLimiter_compMass1_linearY_1, displacingDeltaImpulse_1, body1_displacingVelocityY_1);
        body1_displacingAngularVelocity_1 = _mm256_fmadd_ps(j_normalLimiter_compMass1_angular_1, displacingDeltaImpulse_1, body1_displacingAngularVelocity_1);

        body2_displacingVelocityX_1 = _mm256_fmadd_ps(j_normalLimiter_compMass2_linearX_1, displacingDeltaImpulse_1, body2_displacingVelocityX_1);
        body2_displacingVelocityY_1 = _mm256_fmadd_ps(j_normalLimiter_compMass2_linearY_1, displacingDeltaImpulse_1, body2_displacingVelocityY_1);
        body2_displacingAngularVelocity_1 = _mm256_fmadd_ps(j_normalLimiter_compMass2_angular_1, displacingDeltaImpulse_1, body2_displacingAngularVelocity_1);

        j_normalLimiter_accumulatedDisplacingImpulse_1 = _mm256_add_ps(j_normalLimiter_accumulatedDisplacingImpulse_1, displacingDeltaImpulse_1);

        _mm256_store_ps(&jointP.normalLimiter_accumulatedDisplacingImpulse[iP_0], j_normalLimiter_accumulatedDisplacingImpulse_0);
        _mm256_store_ps(&jointP.normalLimiter_accumulatedDisplacingImpulse[iP_1], j_normalLimiter_accumulatedDisplacingImpulse_1);

        V8f productive_0 = _mm256_cmp_ps(_mm256_andnot_ps(sign, displacingDeltaImpulse_0), _mm256_set1_ps(kProductiveImpulse), _CMP_GT_OQ);
        V8f productive_1 = _mm256_cmp_ps(_mm256_andnot_ps(sign, displacingDeltaImpulse_1), _mm256_set1_ps(kProductiveImpulse), _CMP_GT_OQ);

        productive_any = _mm256_or_si256(productive_any, _mm256_or_si256(productive_0, productive_1));

        body1_lastDisplacementIteration_0 = _mm256_blendv_epi8(body1_lastDisplacementIteration_0, iterationIndex0, productive_0);
        body2_lastDisplacementIteration_0 = _mm256_blendv_epi8(body2_lastDisplacementIteration_0, iterationIndex0, productive_0);

        body1_lastDisplacementIteration_1 = _mm256_blendv_epi8(body1_lastDisplacementIteration_1, iterationIndex0, productive_1);
        body2_lastDisplacementIteration_1 = _mm256_blendv_epi8(body2_lastDisplacementIteration_1, iterationIndex0, productive_1);

        // this is a bit painful :(
        static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Storing assumes fixed layout");

        row0 = body1_displacingVelocityX_0;
        row1 = body1_displacingVelocityY_0;
        row2 = body1_displacingAngularVelocity_0;
        row3 = bitcast(body1_lastDisplacementIteration_0);

        row4 = body2_displacingVelocityX_0;
        row5 = body2_displacingVelocityY_0;
        row6 = body2_displacingAngularVelocity_0;
        row7 = bitcast(body2_lastDisplacementIteration_0);

        _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 0]].velocity.x, _mm256_extractf128_ps(row0, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_0 + 0]].velocity.x, _mm256_extractf128_ps(row0, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 1]].velocity.x, _mm256_extractf128_ps(row1, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_0 + 1]].velocity.x, _mm256_extractf128_ps(row1, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 2]].velocity.x, _mm256_extractf128_ps(row2, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_0 + 2]].velocity.x, _mm256_extractf128_ps(row2, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 3]].velocity.x, _mm256_extractf128_ps(row3, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_0 + 3]].velocity.x, _mm256_extractf128_ps(row3, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 4]].velocity.x, _mm256_extractf128_ps(row4, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_0 + 4]].velocity.x, _mm256_extractf128_ps(row4, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 5]].velocity.x, _mm256_extractf128_ps(row5, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_0 + 5]].velocity.x, _mm256_extractf128_ps(row5, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 6]].velocity.x, _mm256_extractf128_ps(row6, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_0 + 6]].velocity.x, _mm256_extractf128_ps(row6, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_0 + 7]].velocity.x, _mm256_extractf128_ps(row7, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_0 + 7]].velocity.x, _mm256_extractf128_ps(row7, 1));

        row0 = body1_displacingVelocityX_1;
        row1 = body1_displacingVelocityY_1;
        row2 = body1_displacingAngularVelocity_1;
        row3 = bitcast(body1_lastDisplacementIteration_1);

        row4 = body2_displacingVelocityX_1;
        row5 = body2_displacingVelocityY_1;
        row6 = body2_displacingAngularVelocity_1;
        row7 = bitcast(body2_lastDisplacementIteration_1);

        _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 0]].velocity.x, _mm256_extractf128_ps(row0, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_1 + 0]].velocity.x, _mm256_extractf128_ps(row0, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 1]].velocity.x, _mm256_extractf128_ps(row1, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_1 + 1]].velocity.x, _mm256_extractf128_ps(row1, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 2]].velocity.x, _mm256_extractf128_ps(row2, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_1 + 2]].velocity.x, _mm256_extractf128_ps(row2, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 3]].velocity.x, _mm256_extractf128_ps(row3, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_1 + 3]].velocity.x, _mm256_extractf128_ps(row3, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 4]].velocity.x, _mm256_extractf128_ps(row4, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_1 + 4]].velocity.x, _mm256_extractf128_ps(row4, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 5]].velocity.x, _mm256_extractf128_ps(row5, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_1 + 5]].velocity.x, _mm256_extractf128_ps(row5, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 6]].velocity.x, _mm256_extractf128_ps(row6, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_1 + 6]].velocity.x, _mm256_extractf128_ps(row6, 1));

        _mm_store_ps(&solveBodiesDisplacement[jointP.body1Index[iP_1 + 7]].velocity.x, _mm256_extractf128_ps(row7, 0));
        _mm_store_ps(&solveBodiesDisplacement[jointP.body2Index[iP_1 + 7]].velocity.x, _mm256_extractf128_ps(row7, 1));
    }

    return _mm256_movemask_epi8(productive_any) != 0;
}
#endif