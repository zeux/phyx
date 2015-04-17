#include "Joints.h"
#include <assert.h>
#include <vector>

#include "Parallel.h"

#include <immintrin.h>

const float kProductiveImpulse = 1e-4f;

template <typename T>
struct AlignedArray
{
    T* data;
    int size;
    int capacity;

    AlignedArray()
        : data(0)
        , size(0)
        , capacity(0)
    {
    }

    ~AlignedArray()
    {
        aligned_free(data);
    }

    T& operator[](int i)
    {
        return data[i];
    }

    void resize(int newsize)
    {
        if (newsize > capacity)
        {
            int newcapacity = capacity;
            while (newcapacity < newsize)
                newcapacity += newcapacity / 2 + 1;

            aligned_free(data);

            data = static_cast<T*>(aligned_alloc(newcapacity * sizeof(T), 32));
            capacity = newcapacity;
        }

        size = newsize;
    }

    static void* aligned_alloc(size_t size, size_t align)
    {
#ifdef _MSC_VER
        return _aligned_malloc(size, align);
#else
        void* result = 0;
        posix_memalign(&result, align, size);
        return result;
#endif
    }

    static void aligned_free(void* ptr)
    {
#ifdef _MSC_VER
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
};

template <int N>
struct ContactJointPacked
{
    unsigned int body1Index[N];
    unsigned int body2Index[N];

    float normalLimiter_normalProjector1X[N];
    float normalLimiter_normalProjector1Y[N];
    float normalLimiter_normalProjector2X[N];
    float normalLimiter_normalProjector2Y[N];
    float normalLimiter_angularProjector1[N];
    float normalLimiter_angularProjector2[N];

    float normalLimiter_compMass1_linearX[N];
    float normalLimiter_compMass1_linearY[N];
    float normalLimiter_compMass2_linearX[N];
    float normalLimiter_compMass2_linearY[N];
    float normalLimiter_compMass1_angular[N];
    float normalLimiter_compMass2_angular[N];
    float normalLimiter_compInvMass[N];
    float normalLimiter_accumulatedImpulse[N];

    float normalLimiter_dstVelocity[N];
    float normalLimiter_dstDisplacingVelocity[N];
    float normalLimiter_accumulatedDisplacingImpulse[N];

    float frictionLimiter_normalProjector1X[N];
    float frictionLimiter_normalProjector1Y[N];
    float frictionLimiter_normalProjector2X[N];
    float frictionLimiter_normalProjector2Y[N];
    float frictionLimiter_angularProjector1[N];
    float frictionLimiter_angularProjector2[N];

    float frictionLimiter_compMass1_linearX[N];
    float frictionLimiter_compMass1_linearY[N];
    float frictionLimiter_compMass2_linearX[N];
    float frictionLimiter_compMass2_linearY[N];
    float frictionLimiter_compMass1_angular[N];
    float frictionLimiter_compMass2_angular[N];
    float frictionLimiter_compInvMass[N];
    float frictionLimiter_accumulatedImpulse[N];
};

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

struct Solver
{
    Solver()
    {
    }

    NOINLINE void PreStepJoints(WorkQueue& queue)
    {
        ParallelFor(queue, contactJoints.data(), contactJoints.size(), 8, [](ContactJoint& j, int)
                    {
            j.Refresh();
            j.PreStep();
                    });
    }

    NOINLINE float SolveJoints(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
    {
        for (int bodyIndex = 0; bodyIndex < bodiesCount; ++bodyIndex)
        {
            bodies[bodyIndex].lastIteration = -1;
            bodies[bodyIndex].lastDisplacementIteration = -1;
        }

        for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
        {
            for (size_t jointIndex = 0; jointIndex < contactJoints.size(); jointIndex++)
            {
                ContactJoint& joint = contactJoints[jointIndex];

                if (joint.body1->lastIteration < iterationIndex - 1 &&
                    joint.body2->lastIteration < iterationIndex - 1)
                    continue;

                if (joint.SolveImpulse() > kProductiveImpulse)
                {
                    joint.body1->lastIteration = iterationIndex;
                    joint.body2->lastIteration = iterationIndex;
                }
            }
        }

        for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
        {
            for (size_t jointIndex = 0; jointIndex < contactJoints.size(); jointIndex++)
            {
                ContactJoint& joint = contactJoints[jointIndex];

                if (joint.body1->lastDisplacementIteration < iterationIndex - 1 &&
                    joint.body2->lastDisplacementIteration < iterationIndex - 1)
                    continue;

                if (joint.SolveDisplacement() > kProductiveImpulse)
                {
                    joint.body1->lastDisplacementIteration = iterationIndex;
                    joint.body2->lastDisplacementIteration = iterationIndex;
                }
            }
        }

        int iterationSum = 0;

        for (size_t jointIndex = 0; jointIndex < contactJoints.size(); jointIndex++)
        {
            ContactJoint& joint = contactJoints[jointIndex];

            iterationSum += std::max(joint.body1->lastIteration, joint.body2->lastIteration) + 2;
            iterationSum += std::max(joint.body1->lastDisplacementIteration, joint.body2->lastDisplacementIteration) + 2;
        }

        return float(iterationSum) / float(contactJoints.size());
    }

    NOINLINE float SolveJointsAoS(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
    {
        for (int bodyIndex = 0; bodyIndex < bodiesCount; ++bodyIndex)
        {
            bodies[bodyIndex].lastIteration = -1;
            bodies[bodyIndex].lastDisplacementIteration = -1;
        }

        for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
        {
            SolveJointsImpulsesAoS(0, contactJoints.size(), iterationIndex);
        }

        for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
        {
            SolveJointsDisplacementAoS(0, contactJoints.size(), iterationIndex);
        }

        int iterationSum = 0;

        for (size_t jointIndex = 0; jointIndex < contactJoints.size(); jointIndex++)
        {
            ContactJoint& joint = contactJoints[jointIndex];

            iterationSum += std::max(joint.body1->lastIteration, joint.body2->lastIteration) + 2;
            iterationSum += std::max(joint.body1->lastDisplacementIteration, joint.body2->lastDisplacementIteration) + 2;
        }

        return float(iterationSum) / float(contactJoints.size());
    }

    NOINLINE float SolveJointsSoA_Scalar(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
    {
        int groupOffset = SolvePrepareSoA(bodies, bodiesCount, 1);

        for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
        {
            SolveJointsImpulsesSoA(0, contactJoints.size(), iterationIndex);
        }

        for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
        {
            SolveJointsDisplacementSoA(0, contactJoints.size(), iterationIndex);
        }

        return SolveFinishSoA(bodies, bodiesCount);
    }

    NOINLINE float SolveJointsSoA_SSE2(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
    {
        int groupOffset = SolvePrepareSoA(bodies, bodiesCount, 4);

        for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
        {
            SolveJointsImpulsesSoA_SSE2(0, groupOffset, iterationIndex);
            SolveJointsImpulsesSoA(groupOffset, contactJoints.size() - groupOffset, iterationIndex);
        }

        for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
        {
            SolveJointsDisplacementSoA_SSE2(0, groupOffset, iterationIndex);
            SolveJointsDisplacementSoA(groupOffset, contactJoints.size() - groupOffset, iterationIndex);
        }

        return SolveFinishSoA(bodies, bodiesCount);
    }

#ifdef __AVX2__
    NOINLINE float SolveJointsSoA_AVX2(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
    {
        int groupOffset = SolvePrepareSoA(bodies, bodiesCount, 8);

        for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
        {
            SolveJointsImpulsesSoA_AVX2(0, groupOffset, iterationIndex);
            SolveJointsImpulsesSoA(groupOffset, contactJoints.size() - groupOffset, iterationIndex);
        }

        for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
        {
            SolveJointsDisplacementSoA_AVX2(0, groupOffset, iterationIndex);
            SolveJointsDisplacementSoA(groupOffset, contactJoints.size() - groupOffset, iterationIndex);
        }

        return SolveFinishSoA(bodies, bodiesCount);
    }
#endif

    NOINLINE float SolveJointsSoAPacked_Scalar(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
    {
        int groupOffset = SolvePrepareSoAPacked(joint_packed4, bodies, bodiesCount, 1);

        for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
        {
            SolveJointsImpulsesSoAPacked(joint_packed4.data, 0, contactJoints.size(), iterationIndex);
        }

        for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
        {
            SolveJointsDisplacementSoAPacked(joint_packed4.data, 0, contactJoints.size(), iterationIndex);
        }

        return SolveFinishSoAPacked(joint_packed4, bodies, bodiesCount);
    }

    NOINLINE float SolveJointsSoAPacked_SSE2(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
    {
        int groupOffset = SolvePrepareSoAPacked(joint_packed4, bodies, bodiesCount, 4);

        for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
        {
            SolveJointsImpulsesSoAPacked_SSE2(joint_packed4.data, 0, groupOffset, iterationIndex);
            SolveJointsImpulsesSoAPacked(joint_packed4.data, groupOffset, contactJoints.size() - groupOffset, iterationIndex);
        }

        for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
        {
            SolveJointsDisplacementSoAPacked_SSE2(joint_packed4.data, 0, groupOffset, iterationIndex);
            SolveJointsDisplacementSoAPacked(joint_packed4.data, groupOffset, contactJoints.size() - groupOffset, iterationIndex);
        }

        return SolveFinishSoAPacked(joint_packed4, bodies, bodiesCount);
    }

#ifdef __AVX2__
    NOINLINE float SolveJointsSoAPacked_AVX2(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
    {
        int groupOffset = SolvePrepareSoAPacked(joint_packed8, bodies, bodiesCount, 8);

        for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
        {
            SolveJointsImpulsesSoAPacked_AVX2(joint_packed8.data, 0, groupOffset, iterationIndex);
            SolveJointsImpulsesSoAPacked(joint_packed8.data, groupOffset, contactJoints.size() - groupOffset, iterationIndex);
        }

        for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
        {
            SolveJointsDisplacementSoAPacked_AVX2(joint_packed8.data, 0, groupOffset, iterationIndex);
            SolveJointsDisplacementSoAPacked(joint_packed8.data, groupOffset, contactJoints.size() - groupOffset, iterationIndex);
        }

        return SolveFinishSoAPacked(joint_packed8, bodies, bodiesCount);
    }
#endif

#if defined(__AVX2__) && defined(__FMA__)
    NOINLINE float SolveJointsSoAPacked_FMA(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
    {
        int groupOffset = SolvePrepareSoAPacked(joint_packed16, bodies, bodiesCount, 16);

        for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
        {
            SolveJointsImpulsesSoAPacked_FMA(joint_packed16.data, 0, groupOffset, iterationIndex);
            SolveJointsImpulsesSoAPacked(joint_packed16.data, groupOffset, contactJoints.size() - groupOffset, iterationIndex);
        }

        for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
        {
            SolveJointsDisplacementSoAPacked_FMA(joint_packed16.data, 0, groupOffset, iterationIndex);
            SolveJointsDisplacementSoAPacked(joint_packed16.data, groupOffset, contactJoints.size() - groupOffset, iterationIndex);
        }

        return SolveFinishSoAPacked(joint_packed16, bodies, bodiesCount);
    }
#endif

    NOINLINE int SolvePrepareIndicesSoA(int bodiesCount, int groupSizeTarget)
    {
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

    NOINLINE int SolvePrepareSoA(RigidBody* bodies, int bodiesCount, int groupSizeTarget)
    {
        solveBodies.resize(bodiesCount);

        for (int i = 0; i < bodiesCount; ++i)
        {
            solveBodies[i].velocity = bodies[i].velocity;
            solveBodies[i].angularVelocity = bodies[i].angularVelocity;
            solveBodies[i].lastIteration = -1;

            solveBodies[i].displacingVelocity = bodies[i].displacingVelocity;
            solveBodies[i].displacingAngularVelocity = bodies[i].displacingAngularVelocity;
            solveBodies[i].lastDisplacementIteration = -1;
        }

        int jointCount = contactJoints.size();

        joint_index.resize(jointCount);

        joint_body1Index.resize(jointCount);
        joint_body2Index.resize(jointCount);

        joint_normalLimiter_normalProjector1X.resize(jointCount);
        joint_normalLimiter_normalProjector1Y.resize(jointCount);
        joint_normalLimiter_normalProjector2X.resize(jointCount);
        joint_normalLimiter_normalProjector2Y.resize(jointCount);
        joint_normalLimiter_angularProjector1.resize(jointCount);
        joint_normalLimiter_angularProjector2.resize(jointCount);

        joint_normalLimiter_compMass1_linearX.resize(jointCount);
        joint_normalLimiter_compMass1_linearY.resize(jointCount);
        joint_normalLimiter_compMass2_linearX.resize(jointCount);
        joint_normalLimiter_compMass2_linearY.resize(jointCount);
        joint_normalLimiter_compMass1_angular.resize(jointCount);
        joint_normalLimiter_compMass2_angular.resize(jointCount);
        joint_normalLimiter_compInvMass.resize(jointCount);
        joint_normalLimiter_accumulatedImpulse.resize(jointCount);

        joint_normalLimiter_dstVelocity.resize(jointCount);
        joint_normalLimiter_dstDisplacingVelocity.resize(jointCount);
        joint_normalLimiter_accumulatedDisplacingImpulse.resize(jointCount);

        joint_frictionLimiter_normalProjector1X.resize(jointCount);
        joint_frictionLimiter_normalProjector1Y.resize(jointCount);
        joint_frictionLimiter_normalProjector2X.resize(jointCount);
        joint_frictionLimiter_normalProjector2Y.resize(jointCount);
        joint_frictionLimiter_angularProjector1.resize(jointCount);
        joint_frictionLimiter_angularProjector2.resize(jointCount);

        joint_frictionLimiter_compMass1_linearX.resize(jointCount);
        joint_frictionLimiter_compMass1_linearY.resize(jointCount);
        joint_frictionLimiter_compMass2_linearX.resize(jointCount);
        joint_frictionLimiter_compMass2_linearY.resize(jointCount);
        joint_frictionLimiter_compMass1_angular.resize(jointCount);
        joint_frictionLimiter_compMass2_angular.resize(jointCount);
        joint_frictionLimiter_compInvMass.resize(jointCount);
        joint_frictionLimiter_accumulatedImpulse.resize(jointCount);

        int groupOffset = SolvePrepareIndicesSoA(bodiesCount, groupSizeTarget);

        for (int i = 0; i < jointCount; ++i)
        {
            ContactJoint& joint = contactJoints[joint_index[i]];

            joint_body1Index[i] = joint.body1Index;
            joint_body2Index[i] = joint.body2Index;

            joint_normalLimiter_normalProjector1X[i] = joint.normalLimiter.normalProjector1.x;
            joint_normalLimiter_normalProjector1Y[i] = joint.normalLimiter.normalProjector1.y;
            joint_normalLimiter_normalProjector2X[i] = joint.normalLimiter.normalProjector2.x;
            joint_normalLimiter_normalProjector2Y[i] = joint.normalLimiter.normalProjector2.y;
            joint_normalLimiter_angularProjector1[i] = joint.normalLimiter.angularProjector1;
            joint_normalLimiter_angularProjector2[i] = joint.normalLimiter.angularProjector2;

            joint_normalLimiter_compMass1_linearX[i] = joint.normalLimiter.compMass1_linear.x;
            joint_normalLimiter_compMass1_linearY[i] = joint.normalLimiter.compMass1_linear.y;
            joint_normalLimiter_compMass2_linearX[i] = joint.normalLimiter.compMass2_linear.x;
            joint_normalLimiter_compMass2_linearY[i] = joint.normalLimiter.compMass2_linear.y;
            joint_normalLimiter_compMass1_angular[i] = joint.normalLimiter.compMass1_angular;
            joint_normalLimiter_compMass2_angular[i] = joint.normalLimiter.compMass2_angular;
            joint_normalLimiter_compInvMass[i] = joint.normalLimiter.compInvMass;
            joint_normalLimiter_accumulatedImpulse[i] = joint.normalLimiter.accumulatedImpulse;

            joint_normalLimiter_dstVelocity[i] = joint.normalLimiter.dstVelocity;
            joint_normalLimiter_dstDisplacingVelocity[i] = joint.normalLimiter.dstDisplacingVelocity;
            joint_normalLimiter_accumulatedDisplacingImpulse[i] = joint.normalLimiter.accumulatedDisplacingImpulse;

            joint_frictionLimiter_normalProjector1X[i] = joint.frictionLimiter.normalProjector1.x;
            joint_frictionLimiter_normalProjector1Y[i] = joint.frictionLimiter.normalProjector1.y;
            joint_frictionLimiter_normalProjector2X[i] = joint.frictionLimiter.normalProjector2.x;
            joint_frictionLimiter_normalProjector2Y[i] = joint.frictionLimiter.normalProjector2.y;
            joint_frictionLimiter_angularProjector1[i] = joint.frictionLimiter.angularProjector1;
            joint_frictionLimiter_angularProjector2[i] = joint.frictionLimiter.angularProjector2;

            joint_frictionLimiter_compMass1_linearX[i] = joint.frictionLimiter.compMass1_linear.x;
            joint_frictionLimiter_compMass1_linearY[i] = joint.frictionLimiter.compMass1_linear.y;
            joint_frictionLimiter_compMass2_linearX[i] = joint.frictionLimiter.compMass2_linear.x;
            joint_frictionLimiter_compMass2_linearY[i] = joint.frictionLimiter.compMass2_linear.y;
            joint_frictionLimiter_compMass1_angular[i] = joint.frictionLimiter.compMass1_angular;
            joint_frictionLimiter_compMass2_angular[i] = joint.frictionLimiter.compMass2_angular;
            joint_frictionLimiter_compInvMass[i] = joint.frictionLimiter.compInvMass;
            joint_frictionLimiter_accumulatedImpulse[i] = joint.frictionLimiter.accumulatedImpulse;
        }

        return groupOffset;
    }

    template <int N>
    NOINLINE int SolvePrepareSoAPacked(
        AlignedArray<ContactJointPacked<N>>& joint_packed,
        RigidBody* bodies, int bodiesCount, int groupSizeTarget)
    {
        solveBodies.resize(bodiesCount);

        for (int i = 0; i < bodiesCount; ++i)
        {
            solveBodies[i].velocity = bodies[i].velocity;
            solveBodies[i].angularVelocity = bodies[i].angularVelocity;
            solveBodies[i].lastIteration = -1;

            solveBodies[i].displacingVelocity = bodies[i].displacingVelocity;
            solveBodies[i].displacingAngularVelocity = bodies[i].displacingAngularVelocity;
            solveBodies[i].lastDisplacementIteration = -1;
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

    NOINLINE float SolveFinishSoA(RigidBody* bodies, int bodiesCount)
    {
        for (int i = 0; i < bodiesCount; ++i)
        {
            bodies[i].velocity = solveBodies[i].velocity;
            bodies[i].angularVelocity = solveBodies[i].angularVelocity;

            bodies[i].displacingVelocity = solveBodies[i].displacingVelocity;
            bodies[i].displacingAngularVelocity = solveBodies[i].displacingAngularVelocity;
        }

        int jointCount = contactJoints.size();

        for (int i = 0; i < jointCount; ++i)
        {
            ContactJoint& joint = contactJoints[joint_index[i]];

            joint.normalLimiter.accumulatedImpulse = joint_normalLimiter_accumulatedImpulse[i];
            joint.normalLimiter.accumulatedDisplacingImpulse = joint_normalLimiter_accumulatedDisplacingImpulse[i];
            joint.frictionLimiter.accumulatedImpulse = joint_frictionLimiter_accumulatedImpulse[i];
        }

        int iterationSum = 0;

        for (int i = 0; i < jointCount; ++i)
        {
            const SolveBody& body1 = solveBodies[joint_body1Index[i]];
            const SolveBody& body2 = solveBodies[joint_body2Index[i]];

            iterationSum += std::max(body1.lastIteration, body2.lastIteration) + 2;
            iterationSum += std::max(body1.lastDisplacementIteration, body2.lastDisplacementIteration) + 2;
        }

        return float(iterationSum) / float(jointCount);
    }

    template <int N>
    NOINLINE float SolveFinishSoAPacked(
        AlignedArray<ContactJointPacked<N>>& joint_packed,
        RigidBody* bodies, int bodiesCount)
    {
        for (int i = 0; i < bodiesCount; ++i)
        {
            bodies[i].velocity = solveBodies[i].velocity;
            bodies[i].angularVelocity = solveBodies[i].angularVelocity;

            bodies[i].displacingVelocity = solveBodies[i].displacingVelocity;
            bodies[i].displacingAngularVelocity = solveBodies[i].displacingAngularVelocity;
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

            const SolveBody& body1 = solveBodies[jointP.body1Index[iP]];
            const SolveBody& body2 = solveBodies[jointP.body2Index[iP]];

            iterationSum += std::max(body1.lastIteration, body2.lastIteration) + 2;
            iterationSum += std::max(body1.lastDisplacementIteration, body2.lastDisplacementIteration) + 2;
        }

        return float(iterationSum) / float(jointCount);
    }

    NOINLINE void SolveJointsImpulsesAoS(int jointStart, int jointCount, int iterationIndex)
    {
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
            float frictionCoefficient = 0.3f;

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
            }
        }
    }

    NOINLINE void SolveJointsImpulsesSoA(int jointStart, int jointCount, int iterationIndex)
    {
        for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex++)
        {
            int i = jointIndex;

            SolveBody* body1 = &solveBodies[joint_body1Index[i]];
            SolveBody* body2 = &solveBodies[joint_body2Index[i]];

            if (body1->lastIteration < iterationIndex - 1 && body2->lastIteration < iterationIndex - 1)
                continue;

            float normaldV = joint_normalLimiter_dstVelocity[i];

            normaldV -= joint_normalLimiter_normalProjector1X[i] * body1->velocity.x;
            normaldV -= joint_normalLimiter_normalProjector1Y[i] * body1->velocity.y;
            normaldV -= joint_normalLimiter_angularProjector1[i] * body1->angularVelocity;

            normaldV -= joint_normalLimiter_normalProjector2X[i] * body2->velocity.x;
            normaldV -= joint_normalLimiter_normalProjector2Y[i] * body2->velocity.y;
            normaldV -= joint_normalLimiter_angularProjector2[i] * body2->angularVelocity;

            float normalDeltaImpulse = normaldV * joint_normalLimiter_compInvMass[i];

            if (normalDeltaImpulse + joint_normalLimiter_accumulatedImpulse[i] < 0.0f)
                normalDeltaImpulse = -joint_normalLimiter_accumulatedImpulse[i];

            body1->velocity.x += joint_normalLimiter_compMass1_linearX[i] * normalDeltaImpulse;
            body1->velocity.y += joint_normalLimiter_compMass1_linearY[i] * normalDeltaImpulse;
            body1->angularVelocity += joint_normalLimiter_compMass1_angular[i] * normalDeltaImpulse;
            body2->velocity.x += joint_normalLimiter_compMass2_linearX[i] * normalDeltaImpulse;
            body2->velocity.y += joint_normalLimiter_compMass2_linearY[i] * normalDeltaImpulse;
            body2->angularVelocity += joint_normalLimiter_compMass2_angular[i] * normalDeltaImpulse;

            joint_normalLimiter_accumulatedImpulse[i] += normalDeltaImpulse;

            float frictiondV = 0;

            frictiondV -= joint_frictionLimiter_normalProjector1X[i] * body1->velocity.x;
            frictiondV -= joint_frictionLimiter_normalProjector1Y[i] * body1->velocity.y;
            frictiondV -= joint_frictionLimiter_angularProjector1[i] * body1->angularVelocity;

            frictiondV -= joint_frictionLimiter_normalProjector2X[i] * body2->velocity.x;
            frictiondV -= joint_frictionLimiter_normalProjector2Y[i] * body2->velocity.y;
            frictiondV -= joint_frictionLimiter_angularProjector2[i] * body2->angularVelocity;

            float frictionDeltaImpulse = frictiondV * joint_frictionLimiter_compInvMass[i];

            float reactionForce = joint_normalLimiter_accumulatedImpulse[i];
            float accumulatedImpulse = joint_frictionLimiter_accumulatedImpulse[i];

            float frictionForce = accumulatedImpulse + frictionDeltaImpulse;
            float frictionCoefficient = 0.3f;

            if (fabsf(frictionForce) > (reactionForce * frictionCoefficient))
            {
                float dir = frictionForce > 0.0f ? 1.0f : -1.0f;
                frictionForce = dir * reactionForce * frictionCoefficient;
                frictionDeltaImpulse = frictionForce - accumulatedImpulse;
            }

            joint_frictionLimiter_accumulatedImpulse[i] += frictionDeltaImpulse;

            body1->velocity.x += joint_frictionLimiter_compMass1_linearX[i] * frictionDeltaImpulse;
            body1->velocity.y += joint_frictionLimiter_compMass1_linearY[i] * frictionDeltaImpulse;
            body1->angularVelocity += joint_frictionLimiter_compMass1_angular[i] * frictionDeltaImpulse;

            body2->velocity.x += joint_frictionLimiter_compMass2_linearX[i] * frictionDeltaImpulse;
            body2->velocity.y += joint_frictionLimiter_compMass2_linearY[i] * frictionDeltaImpulse;
            body2->angularVelocity += joint_frictionLimiter_compMass2_angular[i] * frictionDeltaImpulse;

            float cumulativeImpulse = std::max(fabsf(normalDeltaImpulse), fabsf(frictionDeltaImpulse));

            if (cumulativeImpulse > kProductiveImpulse)
            {
                body1->lastIteration = iterationIndex;
                body2->lastIteration = iterationIndex;
            }
        }
    }

    NOINLINE void SolveJointsImpulsesSoA_SSE2(int jointStart, int jointCount, int iterationIndex)
    {
        typedef __m128 Vf;
        typedef __m128i Vi;

        assert(jointStart % 4 == 0 && jointCount % 4 == 0);

        Vf sign = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));

        Vi iterationIndex0 = _mm_set1_epi32(iterationIndex);
        Vi iterationIndex2 = _mm_set1_epi32(iterationIndex - 2);

        for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 4)
        {
            int i = jointIndex;

            Vf zero = _mm_setzero_ps();

            __m128 row0, row1, row2, row3;

            static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Loading assumes fixed layout");

            row0 = _mm_load_ps(&solveBodies[joint_body1Index[i + 0]].velocity.x);
            row1 = _mm_load_ps(&solveBodies[joint_body1Index[i + 1]].velocity.x);
            row2 = _mm_load_ps(&solveBodies[joint_body1Index[i + 2]].velocity.x);
            row3 = _mm_load_ps(&solveBodies[joint_body1Index[i + 3]].velocity.x);

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            Vf body1_velocityX = row0;
            Vf body1_velocityY = row1;
            Vf body1_angularVelocity = row2;
            Vi body1_lastIteration = row3;

            row0 = _mm_load_ps(&solveBodies[joint_body2Index[i + 0]].velocity.x);
            row1 = _mm_load_ps(&solveBodies[joint_body2Index[i + 1]].velocity.x);
            row2 = _mm_load_ps(&solveBodies[joint_body2Index[i + 2]].velocity.x);
            row3 = _mm_load_ps(&solveBodies[joint_body2Index[i + 3]].velocity.x);

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            Vf body2_velocityX = row0;
            Vf body2_velocityY = row1;
            Vf body2_angularVelocity = row2;
            Vi body2_lastIteration = row3;

            Vi body1_productive = _mm_cmpgt_epi32(body1_lastIteration, iterationIndex2);
            Vi body2_productive = _mm_cmpgt_epi32(body2_lastIteration, iterationIndex2);
            Vi body_productive = _mm_or_si128(body1_productive, body2_productive);

            if (_mm_movemask_epi8(body_productive) == 0)
                continue;

            Vf j_normalLimiter_normalProjector1X = _mm_load_ps(&joint_normalLimiter_normalProjector1X[i]);
            Vf j_normalLimiter_normalProjector1Y = _mm_load_ps(&joint_normalLimiter_normalProjector1Y[i]);
            Vf j_normalLimiter_normalProjector2X = _mm_load_ps(&joint_normalLimiter_normalProjector2X[i]);
            Vf j_normalLimiter_normalProjector2Y = _mm_load_ps(&joint_normalLimiter_normalProjector2Y[i]);
            Vf j_normalLimiter_angularProjector1 = _mm_load_ps(&joint_normalLimiter_angularProjector1[i]);
            Vf j_normalLimiter_angularProjector2 = _mm_load_ps(&joint_normalLimiter_angularProjector2[i]);

            Vf j_normalLimiter_compMass1_linearX = _mm_load_ps(&joint_normalLimiter_compMass1_linearX[i]);
            Vf j_normalLimiter_compMass1_linearY = _mm_load_ps(&joint_normalLimiter_compMass1_linearY[i]);
            Vf j_normalLimiter_compMass2_linearX = _mm_load_ps(&joint_normalLimiter_compMass2_linearX[i]);
            Vf j_normalLimiter_compMass2_linearY = _mm_load_ps(&joint_normalLimiter_compMass2_linearY[i]);
            Vf j_normalLimiter_compMass1_angular = _mm_load_ps(&joint_normalLimiter_compMass1_angular[i]);
            Vf j_normalLimiter_compMass2_angular = _mm_load_ps(&joint_normalLimiter_compMass2_angular[i]);
            Vf j_normalLimiter_compInvMass = _mm_load_ps(&joint_normalLimiter_compInvMass[i]);
            Vf j_normalLimiter_accumulatedImpulse = _mm_load_ps(&joint_normalLimiter_accumulatedImpulse[i]);
            Vf j_normalLimiter_dstVelocity = _mm_load_ps(&joint_normalLimiter_dstVelocity[i]);

            Vf j_frictionLimiter_normalProjector1X = _mm_load_ps(&joint_frictionLimiter_normalProjector1X[i]);
            Vf j_frictionLimiter_normalProjector1Y = _mm_load_ps(&joint_frictionLimiter_normalProjector1Y[i]);
            Vf j_frictionLimiter_normalProjector2X = _mm_load_ps(&joint_frictionLimiter_normalProjector2X[i]);
            Vf j_frictionLimiter_normalProjector2Y = _mm_load_ps(&joint_frictionLimiter_normalProjector2Y[i]);
            Vf j_frictionLimiter_angularProjector1 = _mm_load_ps(&joint_frictionLimiter_angularProjector1[i]);
            Vf j_frictionLimiter_angularProjector2 = _mm_load_ps(&joint_frictionLimiter_angularProjector2[i]);

            Vf j_frictionLimiter_compMass1_linearX = _mm_load_ps(&joint_frictionLimiter_compMass1_linearX[i]);
            Vf j_frictionLimiter_compMass1_linearY = _mm_load_ps(&joint_frictionLimiter_compMass1_linearY[i]);
            Vf j_frictionLimiter_compMass2_linearX = _mm_load_ps(&joint_frictionLimiter_compMass2_linearX[i]);
            Vf j_frictionLimiter_compMass2_linearY = _mm_load_ps(&joint_frictionLimiter_compMass2_linearY[i]);
            Vf j_frictionLimiter_compMass1_angular = _mm_load_ps(&joint_frictionLimiter_compMass1_angular[i]);
            Vf j_frictionLimiter_compMass2_angular = _mm_load_ps(&joint_frictionLimiter_compMass2_angular[i]);
            Vf j_frictionLimiter_compInvMass = _mm_load_ps(&joint_frictionLimiter_compInvMass[i]);
            Vf j_frictionLimiter_accumulatedImpulse = _mm_load_ps(&joint_frictionLimiter_accumulatedImpulse[i]);

            Vf normaldV = j_normalLimiter_dstVelocity;

            normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_normalProjector1X, body1_velocityX));
            normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_normalProjector1Y, body1_velocityY));
            normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_angularProjector1, body1_angularVelocity));

            normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_normalProjector2X, body2_velocityX));
            normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_normalProjector2Y, body2_velocityY));
            normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_angularProjector2, body2_angularVelocity));

            Vf normalDeltaImpulse = _mm_mul_ps(normaldV, j_normalLimiter_compInvMass);

            normalDeltaImpulse = _mm_max_ps(normalDeltaImpulse, _mm_xor_ps(sign, j_normalLimiter_accumulatedImpulse));

            body1_velocityX = _mm_add_ps(body1_velocityX, _mm_mul_ps(j_normalLimiter_compMass1_linearX, normalDeltaImpulse));
            body1_velocityY = _mm_add_ps(body1_velocityY, _mm_mul_ps(j_normalLimiter_compMass1_linearY, normalDeltaImpulse));
            body1_angularVelocity = _mm_add_ps(body1_angularVelocity, _mm_mul_ps(j_normalLimiter_compMass1_angular, normalDeltaImpulse));

            body2_velocityX = _mm_add_ps(body2_velocityX, _mm_mul_ps(j_normalLimiter_compMass2_linearX, normalDeltaImpulse));
            body2_velocityY = _mm_add_ps(body2_velocityY, _mm_mul_ps(j_normalLimiter_compMass2_linearY, normalDeltaImpulse));
            body2_angularVelocity = _mm_add_ps(body2_angularVelocity, _mm_mul_ps(j_normalLimiter_compMass2_angular, normalDeltaImpulse));

            j_normalLimiter_accumulatedImpulse = _mm_add_ps(j_normalLimiter_accumulatedImpulse, normalDeltaImpulse);

            Vf frictiondV = zero;

            frictiondV = _mm_sub_ps(frictiondV, _mm_mul_ps(j_frictionLimiter_normalProjector1X, body1_velocityX));
            frictiondV = _mm_sub_ps(frictiondV, _mm_mul_ps(j_frictionLimiter_normalProjector1Y, body1_velocityY));
            frictiondV = _mm_sub_ps(frictiondV, _mm_mul_ps(j_frictionLimiter_angularProjector1, body1_angularVelocity));

            frictiondV = _mm_sub_ps(frictiondV, _mm_mul_ps(j_frictionLimiter_normalProjector2X, body2_velocityX));
            frictiondV = _mm_sub_ps(frictiondV, _mm_mul_ps(j_frictionLimiter_normalProjector2Y, body2_velocityY));
            frictiondV = _mm_sub_ps(frictiondV, _mm_mul_ps(j_frictionLimiter_angularProjector2, body2_angularVelocity));

            Vf frictionDeltaImpulse = _mm_mul_ps(frictiondV, j_frictionLimiter_compInvMass);

            Vf reactionForce = j_normalLimiter_accumulatedImpulse;
            Vf accumulatedImpulse = j_frictionLimiter_accumulatedImpulse;

            Vf frictionForce = _mm_add_ps(accumulatedImpulse, frictionDeltaImpulse);
            Vf reactionForceScaled = _mm_mul_ps(reactionForce, _mm_set1_ps(0.3f));

            Vf frictionForceAbs = _mm_andnot_ps(sign, frictionForce);
            Vf reactionForceScaledSigned = _mm_xor_ps(_mm_and_ps(frictionForce, sign), reactionForceScaled);
            Vf frictionDeltaImpulseAdjusted = _mm_sub_ps(reactionForceScaledSigned, accumulatedImpulse);

            Vf frictionSelector = _mm_cmpgt_ps(frictionForceAbs, reactionForceScaled);

            frictionDeltaImpulse = _mm_or_ps(_mm_andnot_ps(frictionSelector, frictionDeltaImpulse), _mm_and_ps(frictionDeltaImpulseAdjusted, frictionSelector));

            j_frictionLimiter_accumulatedImpulse = _mm_add_ps(j_frictionLimiter_accumulatedImpulse, frictionDeltaImpulse);

            body1_velocityX = _mm_add_ps(body1_velocityX, _mm_mul_ps(j_frictionLimiter_compMass1_linearX, frictionDeltaImpulse));
            body1_velocityY = _mm_add_ps(body1_velocityY, _mm_mul_ps(j_frictionLimiter_compMass1_linearY, frictionDeltaImpulse));
            body1_angularVelocity = _mm_add_ps(body1_angularVelocity, _mm_mul_ps(j_frictionLimiter_compMass1_angular, frictionDeltaImpulse));

            body2_velocityX = _mm_add_ps(body2_velocityX, _mm_mul_ps(j_frictionLimiter_compMass2_linearX, frictionDeltaImpulse));
            body2_velocityY = _mm_add_ps(body2_velocityY, _mm_mul_ps(j_frictionLimiter_compMass2_linearY, frictionDeltaImpulse));
            body2_angularVelocity = _mm_add_ps(body2_angularVelocity, _mm_mul_ps(j_frictionLimiter_compMass2_angular, frictionDeltaImpulse));

            _mm_store_ps(&joint_normalLimiter_accumulatedImpulse[i], j_normalLimiter_accumulatedImpulse);
            _mm_store_ps(&joint_frictionLimiter_accumulatedImpulse[i], j_frictionLimiter_accumulatedImpulse);

            Vf cumulativeImpulse = _mm_max_ps(_mm_andnot_ps(sign, normalDeltaImpulse), _mm_andnot_ps(sign, frictionDeltaImpulse));

            Vf productive = _mm_cmpgt_ps(cumulativeImpulse, _mm_set1_ps(kProductiveImpulse));

            body1_lastIteration = _mm_or_si128(_mm_andnot_si128(productive, body1_lastIteration), _mm_and_si128(iterationIndex0, productive));
            body2_lastIteration = _mm_or_si128(_mm_andnot_si128(productive, body2_lastIteration), _mm_and_si128(iterationIndex0, productive));

            // this is a bit painful :(
            static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Storing assumes fixed layout");

            row0 = body1_velocityX;
            row1 = body1_velocityY;
            row2 = body1_angularVelocity;
            row3 = body1_lastIteration;

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            _mm_store_ps(&solveBodies[joint_body1Index[i + 0]].velocity.x, row0);
            _mm_store_ps(&solveBodies[joint_body1Index[i + 1]].velocity.x, row1);
            _mm_store_ps(&solveBodies[joint_body1Index[i + 2]].velocity.x, row2);
            _mm_store_ps(&solveBodies[joint_body1Index[i + 3]].velocity.x, row3);

            row0 = body2_velocityX;
            row1 = body2_velocityY;
            row2 = body2_angularVelocity;
            row3 = body2_lastIteration;

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            _mm_store_ps(&solveBodies[joint_body2Index[i + 0]].velocity.x, row0);
            _mm_store_ps(&solveBodies[joint_body2Index[i + 1]].velocity.x, row1);
            _mm_store_ps(&solveBodies[joint_body2Index[i + 2]].velocity.x, row2);
            _mm_store_ps(&solveBodies[joint_body2Index[i + 3]].velocity.x, row3);
        }
    }

#ifdef __AVX2__
    NOINLINE void SolveJointsImpulsesSoA_AVX2(int jointStart, int jointCount, int iterationIndex)
    {
        typedef __m256 Vf;
        typedef __m256i Vi;

        assert(jointStart % 8 == 0 && jointCount % 8 == 0);

        Vf sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

        Vi iterationIndex0 = _mm256_set1_epi32(iterationIndex);
        Vi iterationIndex2 = _mm256_set1_epi32(iterationIndex - 2);

        for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 8)
        {
            int i = jointIndex;

            Vf zero = _mm256_setzero_ps();

            Vf row0, row1, row2, row3, row4, row5, row6, row7;

            static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Loading assumes fixed layout");

            row0 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 0]].velocity.x, &solveBodies[joint_body2Index[i + 0]].velocity.x);
            row1 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 1]].velocity.x, &solveBodies[joint_body2Index[i + 1]].velocity.x);
            row2 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 2]].velocity.x, &solveBodies[joint_body2Index[i + 2]].velocity.x);
            row3 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 3]].velocity.x, &solveBodies[joint_body2Index[i + 3]].velocity.x);
            row4 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 4]].velocity.x, &solveBodies[joint_body2Index[i + 4]].velocity.x);
            row5 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 5]].velocity.x, &solveBodies[joint_body2Index[i + 5]].velocity.x);
            row6 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 6]].velocity.x, &solveBodies[joint_body2Index[i + 6]].velocity.x);
            row7 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 7]].velocity.x, &solveBodies[joint_body2Index[i + 7]].velocity.x);

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            Vf body1_velocityX = row0;
            Vf body1_velocityY = row1;
            Vf body1_angularVelocity = row2;
            Vi body1_lastIteration = row3;

            Vf body2_velocityX = row4;
            Vf body2_velocityY = row5;
            Vf body2_angularVelocity = row6;
            Vi body2_lastIteration = row7;

            Vi body_lastIteration = _mm256_max_epi32(body1_lastIteration, body2_lastIteration);
            Vi body_productive = _mm256_cmpgt_epi32(body_lastIteration, iterationIndex2);

            if (_mm256_movemask_epi8(body_productive) == 0)
                continue;

            Vf j_normalLimiter_normalProjector1X = _mm256_load_ps(&joint_normalLimiter_normalProjector1X[i]);
            Vf j_normalLimiter_normalProjector1Y = _mm256_load_ps(&joint_normalLimiter_normalProjector1Y[i]);
            Vf j_normalLimiter_normalProjector2X = _mm256_load_ps(&joint_normalLimiter_normalProjector2X[i]);
            Vf j_normalLimiter_normalProjector2Y = _mm256_load_ps(&joint_normalLimiter_normalProjector2Y[i]);
            Vf j_normalLimiter_angularProjector1 = _mm256_load_ps(&joint_normalLimiter_angularProjector1[i]);
            Vf j_normalLimiter_angularProjector2 = _mm256_load_ps(&joint_normalLimiter_angularProjector2[i]);

            Vf j_normalLimiter_compMass1_linearX = _mm256_load_ps(&joint_normalLimiter_compMass1_linearX[i]);
            Vf j_normalLimiter_compMass1_linearY = _mm256_load_ps(&joint_normalLimiter_compMass1_linearY[i]);
            Vf j_normalLimiter_compMass2_linearX = _mm256_load_ps(&joint_normalLimiter_compMass2_linearX[i]);
            Vf j_normalLimiter_compMass2_linearY = _mm256_load_ps(&joint_normalLimiter_compMass2_linearY[i]);
            Vf j_normalLimiter_compMass1_angular = _mm256_load_ps(&joint_normalLimiter_compMass1_angular[i]);
            Vf j_normalLimiter_compMass2_angular = _mm256_load_ps(&joint_normalLimiter_compMass2_angular[i]);
            Vf j_normalLimiter_compInvMass = _mm256_load_ps(&joint_normalLimiter_compInvMass[i]);
            Vf j_normalLimiter_accumulatedImpulse = _mm256_load_ps(&joint_normalLimiter_accumulatedImpulse[i]);
            Vf j_normalLimiter_dstVelocity = _mm256_load_ps(&joint_normalLimiter_dstVelocity[i]);

            Vf j_frictionLimiter_normalProjector1X = _mm256_load_ps(&joint_frictionLimiter_normalProjector1X[i]);
            Vf j_frictionLimiter_normalProjector1Y = _mm256_load_ps(&joint_frictionLimiter_normalProjector1Y[i]);
            Vf j_frictionLimiter_normalProjector2X = _mm256_load_ps(&joint_frictionLimiter_normalProjector2X[i]);
            Vf j_frictionLimiter_normalProjector2Y = _mm256_load_ps(&joint_frictionLimiter_normalProjector2Y[i]);
            Vf j_frictionLimiter_angularProjector1 = _mm256_load_ps(&joint_frictionLimiter_angularProjector1[i]);
            Vf j_frictionLimiter_angularProjector2 = _mm256_load_ps(&joint_frictionLimiter_angularProjector2[i]);

            Vf j_frictionLimiter_compMass1_linearX = _mm256_load_ps(&joint_frictionLimiter_compMass1_linearX[i]);
            Vf j_frictionLimiter_compMass1_linearY = _mm256_load_ps(&joint_frictionLimiter_compMass1_linearY[i]);
            Vf j_frictionLimiter_compMass2_linearX = _mm256_load_ps(&joint_frictionLimiter_compMass2_linearX[i]);
            Vf j_frictionLimiter_compMass2_linearY = _mm256_load_ps(&joint_frictionLimiter_compMass2_linearY[i]);
            Vf j_frictionLimiter_compMass1_angular = _mm256_load_ps(&joint_frictionLimiter_compMass1_angular[i]);
            Vf j_frictionLimiter_compMass2_angular = _mm256_load_ps(&joint_frictionLimiter_compMass2_angular[i]);
            Vf j_frictionLimiter_compInvMass = _mm256_load_ps(&joint_frictionLimiter_compInvMass[i]);
            Vf j_frictionLimiter_accumulatedImpulse = _mm256_load_ps(&joint_frictionLimiter_accumulatedImpulse[i]);

            Vf normaldV = j_normalLimiter_dstVelocity;

            normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector1X, body1_velocityX));
            normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector1Y, body1_velocityY));
            normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_angularProjector1, body1_angularVelocity));

            normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector2X, body2_velocityX));
            normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector2Y, body2_velocityY));
            normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_angularProjector2, body2_angularVelocity));

            Vf normalDeltaImpulse = _mm256_mul_ps(normaldV, j_normalLimiter_compInvMass);

            normalDeltaImpulse = _mm256_max_ps(normalDeltaImpulse, _mm256_xor_ps(sign, j_normalLimiter_accumulatedImpulse));

            body1_velocityX = _mm256_add_ps(body1_velocityX, _mm256_mul_ps(j_normalLimiter_compMass1_linearX, normalDeltaImpulse));
            body1_velocityY = _mm256_add_ps(body1_velocityY, _mm256_mul_ps(j_normalLimiter_compMass1_linearY, normalDeltaImpulse));
            body1_angularVelocity = _mm256_add_ps(body1_angularVelocity, _mm256_mul_ps(j_normalLimiter_compMass1_angular, normalDeltaImpulse));

            body2_velocityX = _mm256_add_ps(body2_velocityX, _mm256_mul_ps(j_normalLimiter_compMass2_linearX, normalDeltaImpulse));
            body2_velocityY = _mm256_add_ps(body2_velocityY, _mm256_mul_ps(j_normalLimiter_compMass2_linearY, normalDeltaImpulse));
            body2_angularVelocity = _mm256_add_ps(body2_angularVelocity, _mm256_mul_ps(j_normalLimiter_compMass2_angular, normalDeltaImpulse));

            j_normalLimiter_accumulatedImpulse = _mm256_add_ps(j_normalLimiter_accumulatedImpulse, normalDeltaImpulse);

            Vf frictiondV = zero;

            frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_normalProjector1X, body1_velocityX));
            frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_normalProjector1Y, body1_velocityY));
            frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_angularProjector1, body1_angularVelocity));

            frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_normalProjector2X, body2_velocityX));
            frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_normalProjector2Y, body2_velocityY));
            frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_angularProjector2, body2_angularVelocity));

            Vf frictionDeltaImpulse = _mm256_mul_ps(frictiondV, j_frictionLimiter_compInvMass);

            Vf reactionForce = j_normalLimiter_accumulatedImpulse;
            Vf accumulatedImpulse = j_frictionLimiter_accumulatedImpulse;

            Vf frictionForce = _mm256_add_ps(accumulatedImpulse, frictionDeltaImpulse);
            Vf reactionForceScaled = _mm256_mul_ps(reactionForce, _mm256_set1_ps(0.3f));

            Vf frictionForceAbs = _mm256_andnot_ps(sign, frictionForce);
            Vf reactionForceScaledSigned = _mm256_xor_ps(_mm256_and_ps(frictionForce, sign), reactionForceScaled);
            Vf frictionDeltaImpulseAdjusted = _mm256_sub_ps(reactionForceScaledSigned, accumulatedImpulse);

            Vf frictionSelector = _mm256_cmp_ps(frictionForceAbs, reactionForceScaled, _CMP_GT_OQ);

            frictionDeltaImpulse = _mm256_blendv_ps(frictionDeltaImpulse, frictionDeltaImpulseAdjusted, frictionSelector);

            j_frictionLimiter_accumulatedImpulse = _mm256_add_ps(j_frictionLimiter_accumulatedImpulse, frictionDeltaImpulse);

            body1_velocityX = _mm256_add_ps(body1_velocityX, _mm256_mul_ps(j_frictionLimiter_compMass1_linearX, frictionDeltaImpulse));
            body1_velocityY = _mm256_add_ps(body1_velocityY, _mm256_mul_ps(j_frictionLimiter_compMass1_linearY, frictionDeltaImpulse));
            body1_angularVelocity = _mm256_add_ps(body1_angularVelocity, _mm256_mul_ps(j_frictionLimiter_compMass1_angular, frictionDeltaImpulse));

            body2_velocityX = _mm256_add_ps(body2_velocityX, _mm256_mul_ps(j_frictionLimiter_compMass2_linearX, frictionDeltaImpulse));
            body2_velocityY = _mm256_add_ps(body2_velocityY, _mm256_mul_ps(j_frictionLimiter_compMass2_linearY, frictionDeltaImpulse));
            body2_angularVelocity = _mm256_add_ps(body2_angularVelocity, _mm256_mul_ps(j_frictionLimiter_compMass2_angular, frictionDeltaImpulse));

            _mm256_store_ps(&joint_normalLimiter_accumulatedImpulse[i], j_normalLimiter_accumulatedImpulse);
            _mm256_store_ps(&joint_frictionLimiter_accumulatedImpulse[i], j_frictionLimiter_accumulatedImpulse);

            Vf cumulativeImpulse = _mm256_max_ps(_mm256_andnot_ps(sign, normalDeltaImpulse), _mm256_andnot_ps(sign, frictionDeltaImpulse));

            Vf productive = _mm256_cmp_ps(cumulativeImpulse, _mm256_set1_ps(kProductiveImpulse), _CMP_GT_OQ);

            body1_lastIteration = _mm256_blendv_epi8(body1_lastIteration, iterationIndex0, productive);
            body2_lastIteration = _mm256_blendv_epi8(body2_lastIteration, iterationIndex0, productive);

            // this is a bit painful :(
            static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Storing assumes fixed layout");

            row0 = body1_velocityX;
            row1 = body1_velocityY;
            row2 = body1_angularVelocity;
            row3 = body1_lastIteration;

            row4 = body2_velocityX;
            row5 = body2_velocityY;
            row6 = body2_angularVelocity;
            row7 = body2_lastIteration;

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            _mm_store_ps(&solveBodies[joint_body1Index[i + 0]].velocity.x, _mm256_extractf128_ps(row0, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 0]].velocity.x, _mm256_extractf128_ps(row0, 1));

            _mm_store_ps(&solveBodies[joint_body1Index[i + 1]].velocity.x, _mm256_extractf128_ps(row1, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 1]].velocity.x, _mm256_extractf128_ps(row1, 1));

            _mm_store_ps(&solveBodies[joint_body1Index[i + 2]].velocity.x, _mm256_extractf128_ps(row2, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 2]].velocity.x, _mm256_extractf128_ps(row2, 1));

            _mm_store_ps(&solveBodies[joint_body1Index[i + 3]].velocity.x, _mm256_extractf128_ps(row3, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 3]].velocity.x, _mm256_extractf128_ps(row3, 1));

            _mm_store_ps(&solveBodies[joint_body1Index[i + 4]].velocity.x, _mm256_extractf128_ps(row4, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 4]].velocity.x, _mm256_extractf128_ps(row4, 1));

            _mm_store_ps(&solveBodies[joint_body1Index[i + 5]].velocity.x, _mm256_extractf128_ps(row5, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 5]].velocity.x, _mm256_extractf128_ps(row5, 1));

            _mm_store_ps(&solveBodies[joint_body1Index[i + 6]].velocity.x, _mm256_extractf128_ps(row6, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 6]].velocity.x, _mm256_extractf128_ps(row6, 1));

            _mm_store_ps(&solveBodies[joint_body1Index[i + 7]].velocity.x, _mm256_extractf128_ps(row7, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 7]].velocity.x, _mm256_extractf128_ps(row7, 1));
        }
    }
#endif

    NOINLINE void SolveJointsDisplacementAoS(int jointStart, int jointCount, int iterationIndex)
    {
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
            }
        }
    }

    NOINLINE void SolveJointsDisplacementSoA(int jointStart, int jointCount, int iterationIndex)
    {
        for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex++)
        {
            int i = jointIndex;

            SolveBody* body1 = &solveBodies[joint_body1Index[i]];
            SolveBody* body2 = &solveBodies[joint_body2Index[i]];

            if (body1->lastDisplacementIteration < iterationIndex - 1 && body2->lastDisplacementIteration < iterationIndex - 1)
                continue;

            float dV = joint_normalLimiter_dstDisplacingVelocity[i];

            dV -= joint_normalLimiter_normalProjector1X[i] * body1->displacingVelocity.x;
            dV -= joint_normalLimiter_normalProjector1Y[i] * body1->displacingVelocity.y;
            dV -= joint_normalLimiter_angularProjector1[i] * body1->displacingAngularVelocity;

            dV -= joint_normalLimiter_normalProjector2X[i] * body2->displacingVelocity.x;
            dV -= joint_normalLimiter_normalProjector2Y[i] * body2->displacingVelocity.y;
            dV -= joint_normalLimiter_angularProjector2[i] * body2->displacingAngularVelocity;

            float displacingDeltaImpulse = dV * joint_normalLimiter_compInvMass[i];

            if (displacingDeltaImpulse + joint_normalLimiter_accumulatedDisplacingImpulse[i] < 0.0f)
                displacingDeltaImpulse = -joint_normalLimiter_accumulatedDisplacingImpulse[i];

            body1->displacingVelocity.x += joint_normalLimiter_compMass1_linearX[i] * displacingDeltaImpulse;
            body1->displacingVelocity.y += joint_normalLimiter_compMass1_linearY[i] * displacingDeltaImpulse;
            body1->displacingAngularVelocity += joint_normalLimiter_compMass1_angular[i] * displacingDeltaImpulse;

            body2->displacingVelocity.x += joint_normalLimiter_compMass2_linearX[i] * displacingDeltaImpulse;
            body2->displacingVelocity.y += joint_normalLimiter_compMass2_linearY[i] * displacingDeltaImpulse;
            body2->displacingAngularVelocity += joint_normalLimiter_compMass2_angular[i] * displacingDeltaImpulse;

            joint_normalLimiter_accumulatedDisplacingImpulse[i] += displacingDeltaImpulse;

            if (fabsf(displacingDeltaImpulse) > kProductiveImpulse)
            {
                body1->lastDisplacementIteration = iterationIndex;
                body2->lastDisplacementIteration = iterationIndex;
            }
        }
    }

    NOINLINE void SolveJointsDisplacementSoA_SSE2(int jointStart, int jointCount, int iterationIndex)
    {
        typedef __m128 Vf;
        typedef __m128i Vi;

        assert(jointStart % 4 == 0 && jointCount % 4 == 0);

        Vf sign = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));

        Vi iterationIndex0 = _mm_set1_epi32(iterationIndex);
        Vi iterationIndex2 = _mm_set1_epi32(iterationIndex - 2);

        for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 4)
        {
            int i = jointIndex;

            Vf zero = _mm_setzero_ps();

            __m128 row0, row1, row2, row3;

            static_assert(offsetof(SolveBody, displacingVelocity) == 16 && offsetof(SolveBody, displacingAngularVelocity) == 24, "Loading assumes fixed layout");

            row0 = _mm_load_ps(&solveBodies[joint_body1Index[i + 0]].displacingVelocity.x);
            row1 = _mm_load_ps(&solveBodies[joint_body1Index[i + 1]].displacingVelocity.x);
            row2 = _mm_load_ps(&solveBodies[joint_body1Index[i + 2]].displacingVelocity.x);
            row3 = _mm_load_ps(&solveBodies[joint_body1Index[i + 3]].displacingVelocity.x);

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            Vf body1_displacingVelocityX = row0;
            Vf body1_displacingVelocityY = row1;
            Vf body1_displacingAngularVelocity = row2;
            Vi body1_lastDisplacementIteration = row3;

            row0 = _mm_load_ps(&solveBodies[joint_body2Index[i + 0]].displacingVelocity.x);
            row1 = _mm_load_ps(&solveBodies[joint_body2Index[i + 1]].displacingVelocity.x);
            row2 = _mm_load_ps(&solveBodies[joint_body2Index[i + 2]].displacingVelocity.x);
            row3 = _mm_load_ps(&solveBodies[joint_body2Index[i + 3]].displacingVelocity.x);

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            Vf body2_displacingVelocityX = row0;
            Vf body2_displacingVelocityY = row1;
            Vf body2_displacingAngularVelocity = row2;
            Vi body2_lastDisplacementIteration = row3;

            Vi body1_productive = _mm_cmpgt_epi32(body1_lastDisplacementIteration, iterationIndex2);
            Vi body2_productive = _mm_cmpgt_epi32(body2_lastDisplacementIteration, iterationIndex2);
            Vi body_productive = _mm_or_si128(body1_productive, body2_productive);

            if (_mm_movemask_epi8(body_productive) == 0)
                continue;

            Vf j_normalLimiter_normalProjector1X = _mm_load_ps(&joint_normalLimiter_normalProjector1X[i]);
            Vf j_normalLimiter_normalProjector1Y = _mm_load_ps(&joint_normalLimiter_normalProjector1Y[i]);
            Vf j_normalLimiter_normalProjector2X = _mm_load_ps(&joint_normalLimiter_normalProjector2X[i]);
            Vf j_normalLimiter_normalProjector2Y = _mm_load_ps(&joint_normalLimiter_normalProjector2Y[i]);
            Vf j_normalLimiter_angularProjector1 = _mm_load_ps(&joint_normalLimiter_angularProjector1[i]);
            Vf j_normalLimiter_angularProjector2 = _mm_load_ps(&joint_normalLimiter_angularProjector2[i]);

            Vf j_normalLimiter_compMass1_linearX = _mm_load_ps(&joint_normalLimiter_compMass1_linearX[i]);
            Vf j_normalLimiter_compMass1_linearY = _mm_load_ps(&joint_normalLimiter_compMass1_linearY[i]);
            Vf j_normalLimiter_compMass2_linearX = _mm_load_ps(&joint_normalLimiter_compMass2_linearX[i]);
            Vf j_normalLimiter_compMass2_linearY = _mm_load_ps(&joint_normalLimiter_compMass2_linearY[i]);
            Vf j_normalLimiter_compMass1_angular = _mm_load_ps(&joint_normalLimiter_compMass1_angular[i]);
            Vf j_normalLimiter_compMass2_angular = _mm_load_ps(&joint_normalLimiter_compMass2_angular[i]);
            Vf j_normalLimiter_compInvMass = _mm_load_ps(&joint_normalLimiter_compInvMass[i]);
            Vf j_normalLimiter_dstDisplacingVelocity = _mm_load_ps(&joint_normalLimiter_dstDisplacingVelocity[i]);
            Vf j_normalLimiter_accumulatedDisplacingImpulse = _mm_load_ps(&joint_normalLimiter_accumulatedDisplacingImpulse[i]);

            Vf dV = j_normalLimiter_dstDisplacingVelocity;

            dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_normalProjector1X, body1_displacingVelocityX));
            dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_normalProjector1Y, body1_displacingVelocityY));
            dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_angularProjector1, body1_displacingAngularVelocity));

            dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_normalProjector2X, body2_displacingVelocityX));
            dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_normalProjector2Y, body2_displacingVelocityY));
            dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_angularProjector2, body2_displacingAngularVelocity));

            Vf displacingDeltaImpulse = _mm_mul_ps(dV, j_normalLimiter_compInvMass);

            displacingDeltaImpulse = _mm_max_ps(displacingDeltaImpulse, _mm_xor_ps(sign, j_normalLimiter_accumulatedDisplacingImpulse));

            body1_displacingVelocityX = _mm_add_ps(body1_displacingVelocityX, _mm_mul_ps(j_normalLimiter_compMass1_linearX, displacingDeltaImpulse));
            body1_displacingVelocityY = _mm_add_ps(body1_displacingVelocityY, _mm_mul_ps(j_normalLimiter_compMass1_linearY, displacingDeltaImpulse));
            body1_displacingAngularVelocity = _mm_add_ps(body1_displacingAngularVelocity, _mm_mul_ps(j_normalLimiter_compMass1_angular, displacingDeltaImpulse));

            body2_displacingVelocityX = _mm_add_ps(body2_displacingVelocityX, _mm_mul_ps(j_normalLimiter_compMass2_linearX, displacingDeltaImpulse));
            body2_displacingVelocityY = _mm_add_ps(body2_displacingVelocityY, _mm_mul_ps(j_normalLimiter_compMass2_linearY, displacingDeltaImpulse));
            body2_displacingAngularVelocity = _mm_add_ps(body2_displacingAngularVelocity, _mm_mul_ps(j_normalLimiter_compMass2_angular, displacingDeltaImpulse));

            j_normalLimiter_accumulatedDisplacingImpulse = _mm_add_ps(j_normalLimiter_accumulatedDisplacingImpulse, displacingDeltaImpulse);

            _mm_store_ps(&joint_normalLimiter_accumulatedDisplacingImpulse[i], j_normalLimiter_accumulatedDisplacingImpulse);

            Vf productive = _mm_cmpgt_ps(_mm_andnot_ps(sign, displacingDeltaImpulse), _mm_set1_ps(kProductiveImpulse));

            body1_lastDisplacementIteration = _mm_or_si128(_mm_andnot_si128(productive, body1_lastDisplacementIteration), _mm_and_si128(iterationIndex0, productive));
            body2_lastDisplacementIteration = _mm_or_si128(_mm_andnot_si128(productive, body2_lastDisplacementIteration), _mm_and_si128(iterationIndex0, productive));

            // this is a bit painful :(
            static_assert(offsetof(SolveBody, displacingVelocity) == 16 && offsetof(SolveBody, displacingAngularVelocity) == 24, "Storing assumes fixed layout");

            row0 = body1_displacingVelocityX;
            row1 = body1_displacingVelocityY;
            row2 = body1_displacingAngularVelocity;
            row3 = body1_lastDisplacementIteration;

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            _mm_store_ps(&solveBodies[joint_body1Index[i + 0]].displacingVelocity.x, row0);
            _mm_store_ps(&solveBodies[joint_body1Index[i + 1]].displacingVelocity.x, row1);
            _mm_store_ps(&solveBodies[joint_body1Index[i + 2]].displacingVelocity.x, row2);
            _mm_store_ps(&solveBodies[joint_body1Index[i + 3]].displacingVelocity.x, row3);

            row0 = body2_displacingVelocityX;
            row1 = body2_displacingVelocityY;
            row2 = body2_displacingAngularVelocity;
            row3 = body2_lastDisplacementIteration;

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            _mm_store_ps(&solveBodies[joint_body2Index[i + 0]].displacingVelocity.x, row0);
            _mm_store_ps(&solveBodies[joint_body2Index[i + 1]].displacingVelocity.x, row1);
            _mm_store_ps(&solveBodies[joint_body2Index[i + 2]].displacingVelocity.x, row2);
            _mm_store_ps(&solveBodies[joint_body2Index[i + 3]].displacingVelocity.x, row3);
        }
    }

#ifdef __AVX2__
    NOINLINE void SolveJointsDisplacementSoA_AVX2(int jointStart, int jointCount, int iterationIndex)
    {
        typedef __m256 Vf;
        typedef __m256i Vi;

        assert(jointStart % 8 == 0 && jointCount % 8 == 0);

        Vf sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

        Vi iterationIndex0 = _mm256_set1_epi32(iterationIndex);
        Vi iterationIndex2 = _mm256_set1_epi32(iterationIndex - 2);

        for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 8)
        {
            int i = jointIndex;

            Vf zero = _mm256_setzero_ps();

            Vf row0, row1, row2, row3, row4, row5, row6, row7;

            static_assert(offsetof(SolveBody, displacingVelocity) == 16 && offsetof(SolveBody, displacingAngularVelocity) == 24, "Loading assumes fixed layout");

            row0 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 0]].displacingVelocity.x, &solveBodies[joint_body2Index[i + 0]].displacingVelocity.x);
            row1 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 1]].displacingVelocity.x, &solveBodies[joint_body2Index[i + 1]].displacingVelocity.x);
            row2 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 2]].displacingVelocity.x, &solveBodies[joint_body2Index[i + 2]].displacingVelocity.x);
            row3 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 3]].displacingVelocity.x, &solveBodies[joint_body2Index[i + 3]].displacingVelocity.x);
            row4 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 4]].displacingVelocity.x, &solveBodies[joint_body2Index[i + 4]].displacingVelocity.x);
            row5 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 5]].displacingVelocity.x, &solveBodies[joint_body2Index[i + 5]].displacingVelocity.x);
            row6 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 6]].displacingVelocity.x, &solveBodies[joint_body2Index[i + 6]].displacingVelocity.x);
            row7 = _mm256_load2_m128(&solveBodies[joint_body1Index[i + 7]].displacingVelocity.x, &solveBodies[joint_body2Index[i + 7]].displacingVelocity.x);

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            Vf body1_displacingVelocityX = row0;
            Vf body1_displacingVelocityY = row1;
            Vf body1_displacingAngularVelocity = row2;
            Vi body1_lastDisplacementIteration = row3;

            Vf body2_displacingVelocityX = row4;
            Vf body2_displacingVelocityY = row5;
            Vf body2_displacingAngularVelocity = row6;
            Vi body2_lastDisplacementIteration = row7;

            Vi body_lastDisplacementIteration = _mm256_max_epi32(body1_lastDisplacementIteration, body2_lastDisplacementIteration);
            Vi body_productive = _mm256_cmpgt_epi32(body_lastDisplacementIteration, iterationIndex2);

            if (_mm256_movemask_epi8(body_productive) == 0)
                continue;

            Vf j_normalLimiter_normalProjector1X = _mm256_load_ps(&joint_normalLimiter_normalProjector1X[i]);
            Vf j_normalLimiter_normalProjector1Y = _mm256_load_ps(&joint_normalLimiter_normalProjector1Y[i]);
            Vf j_normalLimiter_normalProjector2X = _mm256_load_ps(&joint_normalLimiter_normalProjector2X[i]);
            Vf j_normalLimiter_normalProjector2Y = _mm256_load_ps(&joint_normalLimiter_normalProjector2Y[i]);
            Vf j_normalLimiter_angularProjector1 = _mm256_load_ps(&joint_normalLimiter_angularProjector1[i]);
            Vf j_normalLimiter_angularProjector2 = _mm256_load_ps(&joint_normalLimiter_angularProjector2[i]);

            Vf j_normalLimiter_compMass1_linearX = _mm256_load_ps(&joint_normalLimiter_compMass1_linearX[i]);
            Vf j_normalLimiter_compMass1_linearY = _mm256_load_ps(&joint_normalLimiter_compMass1_linearY[i]);
            Vf j_normalLimiter_compMass2_linearX = _mm256_load_ps(&joint_normalLimiter_compMass2_linearX[i]);
            Vf j_normalLimiter_compMass2_linearY = _mm256_load_ps(&joint_normalLimiter_compMass2_linearY[i]);
            Vf j_normalLimiter_compMass1_angular = _mm256_load_ps(&joint_normalLimiter_compMass1_angular[i]);
            Vf j_normalLimiter_compMass2_angular = _mm256_load_ps(&joint_normalLimiter_compMass2_angular[i]);
            Vf j_normalLimiter_compInvMass = _mm256_load_ps(&joint_normalLimiter_compInvMass[i]);
            Vf j_normalLimiter_dstDisplacingVelocity = _mm256_load_ps(&joint_normalLimiter_dstDisplacingVelocity[i]);
            Vf j_normalLimiter_accumulatedDisplacingImpulse = _mm256_load_ps(&joint_normalLimiter_accumulatedDisplacingImpulse[i]);

            Vf dV = j_normalLimiter_dstDisplacingVelocity;

            dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector1X, body1_displacingVelocityX));
            dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector1Y, body1_displacingVelocityY));
            dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_angularProjector1, body1_displacingAngularVelocity));

            dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector2X, body2_displacingVelocityX));
            dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector2Y, body2_displacingVelocityY));
            dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_angularProjector2, body2_displacingAngularVelocity));

            Vf displacingDeltaImpulse = _mm256_mul_ps(dV, j_normalLimiter_compInvMass);

            displacingDeltaImpulse = _mm256_max_ps(displacingDeltaImpulse, _mm256_xor_ps(sign, j_normalLimiter_accumulatedDisplacingImpulse));

            body1_displacingVelocityX = _mm256_add_ps(body1_displacingVelocityX, _mm256_mul_ps(j_normalLimiter_compMass1_linearX, displacingDeltaImpulse));
            body1_displacingVelocityY = _mm256_add_ps(body1_displacingVelocityY, _mm256_mul_ps(j_normalLimiter_compMass1_linearY, displacingDeltaImpulse));
            body1_displacingAngularVelocity = _mm256_add_ps(body1_displacingAngularVelocity, _mm256_mul_ps(j_normalLimiter_compMass1_angular, displacingDeltaImpulse));

            body2_displacingVelocityX = _mm256_add_ps(body2_displacingVelocityX, _mm256_mul_ps(j_normalLimiter_compMass2_linearX, displacingDeltaImpulse));
            body2_displacingVelocityY = _mm256_add_ps(body2_displacingVelocityY, _mm256_mul_ps(j_normalLimiter_compMass2_linearY, displacingDeltaImpulse));
            body2_displacingAngularVelocity = _mm256_add_ps(body2_displacingAngularVelocity, _mm256_mul_ps(j_normalLimiter_compMass2_angular, displacingDeltaImpulse));

            j_normalLimiter_accumulatedDisplacingImpulse = _mm256_add_ps(j_normalLimiter_accumulatedDisplacingImpulse, displacingDeltaImpulse);

            _mm256_store_ps(&joint_normalLimiter_accumulatedDisplacingImpulse[i], j_normalLimiter_accumulatedDisplacingImpulse);

            Vf productive = _mm256_cmp_ps(_mm256_andnot_ps(sign, displacingDeltaImpulse), _mm256_set1_ps(kProductiveImpulse), _CMP_GT_OQ);

            body1_lastDisplacementIteration = _mm256_blendv_epi8(body1_lastDisplacementIteration, iterationIndex0, productive);
            body2_lastDisplacementIteration = _mm256_blendv_epi8(body2_lastDisplacementIteration, iterationIndex0, productive);

            // this is a bit painful :(
            static_assert(offsetof(SolveBody, displacingVelocity) == 16 && offsetof(SolveBody, displacingAngularVelocity) == 24, "Storing assumes fixed layout");

            row0 = body1_displacingVelocityX;
            row1 = body1_displacingVelocityY;
            row2 = body1_displacingAngularVelocity;
            row3 = body1_lastDisplacementIteration;

            row4 = body2_displacingVelocityX;
            row5 = body2_displacingVelocityY;
            row6 = body2_displacingAngularVelocity;
            row7 = body2_lastDisplacementIteration;

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            _mm_store_ps(&solveBodies[joint_body1Index[i + 0]].displacingVelocity.x, _mm256_extractf128_ps(row0, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 0]].displacingVelocity.x, _mm256_extractf128_ps(row0, 1));

            _mm_store_ps(&solveBodies[joint_body1Index[i + 1]].displacingVelocity.x, _mm256_extractf128_ps(row1, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 1]].displacingVelocity.x, _mm256_extractf128_ps(row1, 1));

            _mm_store_ps(&solveBodies[joint_body1Index[i + 2]].displacingVelocity.x, _mm256_extractf128_ps(row2, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 2]].displacingVelocity.x, _mm256_extractf128_ps(row2, 1));

            _mm_store_ps(&solveBodies[joint_body1Index[i + 3]].displacingVelocity.x, _mm256_extractf128_ps(row3, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 3]].displacingVelocity.x, _mm256_extractf128_ps(row3, 1));

            _mm_store_ps(&solveBodies[joint_body1Index[i + 4]].displacingVelocity.x, _mm256_extractf128_ps(row4, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 4]].displacingVelocity.x, _mm256_extractf128_ps(row4, 1));

            _mm_store_ps(&solveBodies[joint_body1Index[i + 5]].displacingVelocity.x, _mm256_extractf128_ps(row5, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 5]].displacingVelocity.x, _mm256_extractf128_ps(row5, 1));

            _mm_store_ps(&solveBodies[joint_body1Index[i + 6]].displacingVelocity.x, _mm256_extractf128_ps(row6, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 6]].displacingVelocity.x, _mm256_extractf128_ps(row6, 1));

            _mm_store_ps(&solveBodies[joint_body1Index[i + 7]].displacingVelocity.x, _mm256_extractf128_ps(row7, 0));
            _mm_store_ps(&solveBodies[joint_body2Index[i + 7]].displacingVelocity.x, _mm256_extractf128_ps(row7, 1));
        }
    }
#endif

    template <int N>
    NOINLINE void SolveJointsImpulsesSoAPacked(ContactJointPacked<N>* joint_packed, int jointStart, int jointCount, int iterationIndex)
    {
        for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex++)
        {
            int i = jointIndex;

            ContactJointPacked<N>& jointP = joint_packed[unsigned(i) / N];
            int iP = i & (N - 1);

            SolveBody* body1 = &solveBodies[jointP.body1Index[iP]];
            SolveBody* body2 = &solveBodies[jointP.body2Index[iP]];

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
            float frictionCoefficient = 0.3f;

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
            }
        }
    }

    NOINLINE void SolveJointsImpulsesSoAPacked_SSE2(ContactJointPacked<4>* joint_packed, int jointStart, int jointCount, int iterationIndex)
    {
        typedef __m128 Vf;
        typedef __m128i Vi;

        assert(jointStart % 4 == 0 && jointCount % 4 == 0);

        Vf sign = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));

        Vi iterationIndex0 = _mm_set1_epi32(iterationIndex);
        Vi iterationIndex2 = _mm_set1_epi32(iterationIndex - 2);

        for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 4)
        {
            int i = jointIndex;

            ContactJointPacked<4>& jointP = joint_packed[i >> 2];
            int iP = 0;

            Vf zero = _mm_setzero_ps();

            __m128 row0, row1, row2, row3;

            static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Loading assumes fixed layout");

            row0 = _mm_load_ps(&solveBodies[jointP.body1Index[iP + 0]].velocity.x);
            row1 = _mm_load_ps(&solveBodies[jointP.body1Index[iP + 1]].velocity.x);
            row2 = _mm_load_ps(&solveBodies[jointP.body1Index[iP + 2]].velocity.x);
            row3 = _mm_load_ps(&solveBodies[jointP.body1Index[iP + 3]].velocity.x);

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            Vf body1_velocityX = row0;
            Vf body1_velocityY = row1;
            Vf body1_angularVelocity = row2;
            Vi body1_lastIteration = row3;

            row0 = _mm_load_ps(&solveBodies[jointP.body2Index[iP + 0]].velocity.x);
            row1 = _mm_load_ps(&solveBodies[jointP.body2Index[iP + 1]].velocity.x);
            row2 = _mm_load_ps(&solveBodies[jointP.body2Index[iP + 2]].velocity.x);
            row3 = _mm_load_ps(&solveBodies[jointP.body2Index[iP + 3]].velocity.x);

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            Vf body2_velocityX = row0;
            Vf body2_velocityY = row1;
            Vf body2_angularVelocity = row2;
            Vi body2_lastIteration = row3;

            Vi body1_productive = _mm_cmpgt_epi32(body1_lastIteration, iterationIndex2);
            Vi body2_productive = _mm_cmpgt_epi32(body2_lastIteration, iterationIndex2);
            Vi body_productive = _mm_or_si128(body1_productive, body2_productive);

            if (_mm_movemask_epi8(body_productive) == 0)
                continue;

            Vf j_normalLimiter_normalProjector1X = _mm_load_ps(&jointP.normalLimiter_normalProjector1X[iP]);
            Vf j_normalLimiter_normalProjector1Y = _mm_load_ps(&jointP.normalLimiter_normalProjector1Y[iP]);
            Vf j_normalLimiter_normalProjector2X = _mm_load_ps(&jointP.normalLimiter_normalProjector2X[iP]);
            Vf j_normalLimiter_normalProjector2Y = _mm_load_ps(&jointP.normalLimiter_normalProjector2Y[iP]);
            Vf j_normalLimiter_angularProjector1 = _mm_load_ps(&jointP.normalLimiter_angularProjector1[iP]);
            Vf j_normalLimiter_angularProjector2 = _mm_load_ps(&jointP.normalLimiter_angularProjector2[iP]);

            Vf j_normalLimiter_compMass1_linearX = _mm_load_ps(&jointP.normalLimiter_compMass1_linearX[iP]);
            Vf j_normalLimiter_compMass1_linearY = _mm_load_ps(&jointP.normalLimiter_compMass1_linearY[iP]);
            Vf j_normalLimiter_compMass2_linearX = _mm_load_ps(&jointP.normalLimiter_compMass2_linearX[iP]);
            Vf j_normalLimiter_compMass2_linearY = _mm_load_ps(&jointP.normalLimiter_compMass2_linearY[iP]);
            Vf j_normalLimiter_compMass1_angular = _mm_load_ps(&jointP.normalLimiter_compMass1_angular[iP]);
            Vf j_normalLimiter_compMass2_angular = _mm_load_ps(&jointP.normalLimiter_compMass2_angular[iP]);
            Vf j_normalLimiter_compInvMass = _mm_load_ps(&jointP.normalLimiter_compInvMass[iP]);
            Vf j_normalLimiter_accumulatedImpulse = _mm_load_ps(&jointP.normalLimiter_accumulatedImpulse[iP]);
            Vf j_normalLimiter_dstVelocity = _mm_load_ps(&jointP.normalLimiter_dstVelocity[iP]);

            Vf j_frictionLimiter_normalProjector1X = _mm_load_ps(&jointP.frictionLimiter_normalProjector1X[iP]);
            Vf j_frictionLimiter_normalProjector1Y = _mm_load_ps(&jointP.frictionLimiter_normalProjector1Y[iP]);
            Vf j_frictionLimiter_normalProjector2X = _mm_load_ps(&jointP.frictionLimiter_normalProjector2X[iP]);
            Vf j_frictionLimiter_normalProjector2Y = _mm_load_ps(&jointP.frictionLimiter_normalProjector2Y[iP]);
            Vf j_frictionLimiter_angularProjector1 = _mm_load_ps(&jointP.frictionLimiter_angularProjector1[iP]);
            Vf j_frictionLimiter_angularProjector2 = _mm_load_ps(&jointP.frictionLimiter_angularProjector2[iP]);

            Vf j_frictionLimiter_compMass1_linearX = _mm_load_ps(&jointP.frictionLimiter_compMass1_linearX[iP]);
            Vf j_frictionLimiter_compMass1_linearY = _mm_load_ps(&jointP.frictionLimiter_compMass1_linearY[iP]);
            Vf j_frictionLimiter_compMass2_linearX = _mm_load_ps(&jointP.frictionLimiter_compMass2_linearX[iP]);
            Vf j_frictionLimiter_compMass2_linearY = _mm_load_ps(&jointP.frictionLimiter_compMass2_linearY[iP]);
            Vf j_frictionLimiter_compMass1_angular = _mm_load_ps(&jointP.frictionLimiter_compMass1_angular[iP]);
            Vf j_frictionLimiter_compMass2_angular = _mm_load_ps(&jointP.frictionLimiter_compMass2_angular[iP]);
            Vf j_frictionLimiter_compInvMass = _mm_load_ps(&jointP.frictionLimiter_compInvMass[iP]);
            Vf j_frictionLimiter_accumulatedImpulse = _mm_load_ps(&jointP.frictionLimiter_accumulatedImpulse[iP]);

            Vf normaldV = j_normalLimiter_dstVelocity;

            normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_normalProjector1X, body1_velocityX));
            normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_normalProjector1Y, body1_velocityY));
            normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_angularProjector1, body1_angularVelocity));

            normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_normalProjector2X, body2_velocityX));
            normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_normalProjector2Y, body2_velocityY));
            normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_angularProjector2, body2_angularVelocity));

            Vf normalDeltaImpulse = _mm_mul_ps(normaldV, j_normalLimiter_compInvMass);

            normalDeltaImpulse = _mm_max_ps(normalDeltaImpulse, _mm_xor_ps(sign, j_normalLimiter_accumulatedImpulse));

            body1_velocityX = _mm_add_ps(body1_velocityX, _mm_mul_ps(j_normalLimiter_compMass1_linearX, normalDeltaImpulse));
            body1_velocityY = _mm_add_ps(body1_velocityY, _mm_mul_ps(j_normalLimiter_compMass1_linearY, normalDeltaImpulse));
            body1_angularVelocity = _mm_add_ps(body1_angularVelocity, _mm_mul_ps(j_normalLimiter_compMass1_angular, normalDeltaImpulse));

            body2_velocityX = _mm_add_ps(body2_velocityX, _mm_mul_ps(j_normalLimiter_compMass2_linearX, normalDeltaImpulse));
            body2_velocityY = _mm_add_ps(body2_velocityY, _mm_mul_ps(j_normalLimiter_compMass2_linearY, normalDeltaImpulse));
            body2_angularVelocity = _mm_add_ps(body2_angularVelocity, _mm_mul_ps(j_normalLimiter_compMass2_angular, normalDeltaImpulse));

            j_normalLimiter_accumulatedImpulse = _mm_add_ps(j_normalLimiter_accumulatedImpulse, normalDeltaImpulse);

            Vf frictiondV = zero;

            frictiondV = _mm_sub_ps(frictiondV, _mm_mul_ps(j_frictionLimiter_normalProjector1X, body1_velocityX));
            frictiondV = _mm_sub_ps(frictiondV, _mm_mul_ps(j_frictionLimiter_normalProjector1Y, body1_velocityY));
            frictiondV = _mm_sub_ps(frictiondV, _mm_mul_ps(j_frictionLimiter_angularProjector1, body1_angularVelocity));

            frictiondV = _mm_sub_ps(frictiondV, _mm_mul_ps(j_frictionLimiter_normalProjector2X, body2_velocityX));
            frictiondV = _mm_sub_ps(frictiondV, _mm_mul_ps(j_frictionLimiter_normalProjector2Y, body2_velocityY));
            frictiondV = _mm_sub_ps(frictiondV, _mm_mul_ps(j_frictionLimiter_angularProjector2, body2_angularVelocity));

            Vf frictionDeltaImpulse = _mm_mul_ps(frictiondV, j_frictionLimiter_compInvMass);

            Vf reactionForce = j_normalLimiter_accumulatedImpulse;
            Vf accumulatedImpulse = j_frictionLimiter_accumulatedImpulse;

            Vf frictionForce = _mm_add_ps(accumulatedImpulse, frictionDeltaImpulse);
            Vf reactionForceScaled = _mm_mul_ps(reactionForce, _mm_set1_ps(0.3f));

            Vf frictionForceAbs = _mm_andnot_ps(sign, frictionForce);
            Vf reactionForceScaledSigned = _mm_xor_ps(_mm_and_ps(frictionForce, sign), reactionForceScaled);
            Vf frictionDeltaImpulseAdjusted = _mm_sub_ps(reactionForceScaledSigned, accumulatedImpulse);

            Vf frictionSelector = _mm_cmpgt_ps(frictionForceAbs, reactionForceScaled);

            frictionDeltaImpulse = _mm_or_ps(_mm_andnot_ps(frictionSelector, frictionDeltaImpulse), _mm_and_ps(frictionDeltaImpulseAdjusted, frictionSelector));

            j_frictionLimiter_accumulatedImpulse = _mm_add_ps(j_frictionLimiter_accumulatedImpulse, frictionDeltaImpulse);

            body1_velocityX = _mm_add_ps(body1_velocityX, _mm_mul_ps(j_frictionLimiter_compMass1_linearX, frictionDeltaImpulse));
            body1_velocityY = _mm_add_ps(body1_velocityY, _mm_mul_ps(j_frictionLimiter_compMass1_linearY, frictionDeltaImpulse));
            body1_angularVelocity = _mm_add_ps(body1_angularVelocity, _mm_mul_ps(j_frictionLimiter_compMass1_angular, frictionDeltaImpulse));

            body2_velocityX = _mm_add_ps(body2_velocityX, _mm_mul_ps(j_frictionLimiter_compMass2_linearX, frictionDeltaImpulse));
            body2_velocityY = _mm_add_ps(body2_velocityY, _mm_mul_ps(j_frictionLimiter_compMass2_linearY, frictionDeltaImpulse));
            body2_angularVelocity = _mm_add_ps(body2_angularVelocity, _mm_mul_ps(j_frictionLimiter_compMass2_angular, frictionDeltaImpulse));

            _mm_store_ps(&jointP.normalLimiter_accumulatedImpulse[iP], j_normalLimiter_accumulatedImpulse);
            _mm_store_ps(&jointP.frictionLimiter_accumulatedImpulse[iP], j_frictionLimiter_accumulatedImpulse);

            Vf cumulativeImpulse = _mm_max_ps(_mm_andnot_ps(sign, normalDeltaImpulse), _mm_andnot_ps(sign, frictionDeltaImpulse));

            Vf productive = _mm_cmpgt_ps(cumulativeImpulse, _mm_set1_ps(kProductiveImpulse));

            body1_lastIteration = _mm_or_si128(_mm_andnot_si128(productive, body1_lastIteration), _mm_and_si128(iterationIndex0, productive));
            body2_lastIteration = _mm_or_si128(_mm_andnot_si128(productive, body2_lastIteration), _mm_and_si128(iterationIndex0, productive));

            // this is a bit painful :(
            static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Storing assumes fixed layout");

            row0 = body1_velocityX;
            row1 = body1_velocityY;
            row2 = body1_angularVelocity;
            row3 = body1_lastIteration;

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 0]].velocity.x, row0);
            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 1]].velocity.x, row1);
            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 2]].velocity.x, row2);
            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 3]].velocity.x, row3);

            row0 = body2_velocityX;
            row1 = body2_velocityY;
            row2 = body2_angularVelocity;
            row3 = body2_lastIteration;

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 0]].velocity.x, row0);
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 1]].velocity.x, row1);
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 2]].velocity.x, row2);
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 3]].velocity.x, row3);
        }
    }

#ifdef __AVX2__
    NOINLINE void SolveJointsImpulsesSoAPacked_AVX2(ContactJointPacked<8>* joint_packed, int jointStart, int jointCount, int iterationIndex)
    {
        typedef __m256 Vf;
        typedef __m256i Vi;

        assert(jointStart % 8 == 0 && jointCount % 8 == 0);

        Vf sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

        Vi iterationIndex0 = _mm256_set1_epi32(iterationIndex);
        Vi iterationIndex2 = _mm256_set1_epi32(iterationIndex - 2);

        for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 8)
        {
            int i = jointIndex;

            ContactJointPacked<8>& jointP = joint_packed[i >> 3];
            int iP = 0;

            Vf zero = _mm256_setzero_ps();

            Vf row0, row1, row2, row3, row4, row5, row6, row7;

            static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Loading assumes fixed layout");

            row0 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 0]].velocity.x, &solveBodies[jointP.body2Index[iP + 0]].velocity.x);
            row1 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 1]].velocity.x, &solveBodies[jointP.body2Index[iP + 1]].velocity.x);
            row2 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 2]].velocity.x, &solveBodies[jointP.body2Index[iP + 2]].velocity.x);
            row3 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 3]].velocity.x, &solveBodies[jointP.body2Index[iP + 3]].velocity.x);
            row4 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 4]].velocity.x, &solveBodies[jointP.body2Index[iP + 4]].velocity.x);
            row5 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 5]].velocity.x, &solveBodies[jointP.body2Index[iP + 5]].velocity.x);
            row6 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 6]].velocity.x, &solveBodies[jointP.body2Index[iP + 6]].velocity.x);
            row7 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 7]].velocity.x, &solveBodies[jointP.body2Index[iP + 7]].velocity.x);

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            Vf body1_velocityX = row0;
            Vf body1_velocityY = row1;
            Vf body1_angularVelocity = row2;
            Vi body1_lastIteration = row3;

            Vf body2_velocityX = row4;
            Vf body2_velocityY = row5;
            Vf body2_angularVelocity = row6;
            Vi body2_lastIteration = row7;

            Vi body_lastIteration = _mm256_max_epi32(body1_lastIteration, body2_lastIteration);
            Vi body_productive = _mm256_cmpgt_epi32(body_lastIteration, iterationIndex2);

            if (_mm256_movemask_epi8(body_productive) == 0)
                continue;

            Vf j_normalLimiter_normalProjector1X = _mm256_load_ps(&jointP.normalLimiter_normalProjector1X[iP]);
            Vf j_normalLimiter_normalProjector1Y = _mm256_load_ps(&jointP.normalLimiter_normalProjector1Y[iP]);
            Vf j_normalLimiter_normalProjector2X = _mm256_load_ps(&jointP.normalLimiter_normalProjector2X[iP]);
            Vf j_normalLimiter_normalProjector2Y = _mm256_load_ps(&jointP.normalLimiter_normalProjector2Y[iP]);
            Vf j_normalLimiter_angularProjector1 = _mm256_load_ps(&jointP.normalLimiter_angularProjector1[iP]);
            Vf j_normalLimiter_angularProjector2 = _mm256_load_ps(&jointP.normalLimiter_angularProjector2[iP]);

            Vf j_normalLimiter_compMass1_linearX = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearX[iP]);
            Vf j_normalLimiter_compMass1_linearY = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearY[iP]);
            Vf j_normalLimiter_compMass2_linearX = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearX[iP]);
            Vf j_normalLimiter_compMass2_linearY = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearY[iP]);
            Vf j_normalLimiter_compMass1_angular = _mm256_load_ps(&jointP.normalLimiter_compMass1_angular[iP]);
            Vf j_normalLimiter_compMass2_angular = _mm256_load_ps(&jointP.normalLimiter_compMass2_angular[iP]);
            Vf j_normalLimiter_compInvMass = _mm256_load_ps(&jointP.normalLimiter_compInvMass[iP]);
            Vf j_normalLimiter_accumulatedImpulse = _mm256_load_ps(&jointP.normalLimiter_accumulatedImpulse[iP]);
            Vf j_normalLimiter_dstVelocity = _mm256_load_ps(&jointP.normalLimiter_dstVelocity[iP]);

            Vf j_frictionLimiter_normalProjector1X = _mm256_load_ps(&jointP.frictionLimiter_normalProjector1X[iP]);
            Vf j_frictionLimiter_normalProjector1Y = _mm256_load_ps(&jointP.frictionLimiter_normalProjector1Y[iP]);
            Vf j_frictionLimiter_normalProjector2X = _mm256_load_ps(&jointP.frictionLimiter_normalProjector2X[iP]);
            Vf j_frictionLimiter_normalProjector2Y = _mm256_load_ps(&jointP.frictionLimiter_normalProjector2Y[iP]);
            Vf j_frictionLimiter_angularProjector1 = _mm256_load_ps(&jointP.frictionLimiter_angularProjector1[iP]);
            Vf j_frictionLimiter_angularProjector2 = _mm256_load_ps(&jointP.frictionLimiter_angularProjector2[iP]);

            Vf j_frictionLimiter_compMass1_linearX = _mm256_load_ps(&jointP.frictionLimiter_compMass1_linearX[iP]);
            Vf j_frictionLimiter_compMass1_linearY = _mm256_load_ps(&jointP.frictionLimiter_compMass1_linearY[iP]);
            Vf j_frictionLimiter_compMass2_linearX = _mm256_load_ps(&jointP.frictionLimiter_compMass2_linearX[iP]);
            Vf j_frictionLimiter_compMass2_linearY = _mm256_load_ps(&jointP.frictionLimiter_compMass2_linearY[iP]);
            Vf j_frictionLimiter_compMass1_angular = _mm256_load_ps(&jointP.frictionLimiter_compMass1_angular[iP]);
            Vf j_frictionLimiter_compMass2_angular = _mm256_load_ps(&jointP.frictionLimiter_compMass2_angular[iP]);
            Vf j_frictionLimiter_compInvMass = _mm256_load_ps(&jointP.frictionLimiter_compInvMass[iP]);
            Vf j_frictionLimiter_accumulatedImpulse = _mm256_load_ps(&jointP.frictionLimiter_accumulatedImpulse[iP]);

            Vf normaldV = j_normalLimiter_dstVelocity;

            normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector1X, body1_velocityX));
            normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector1Y, body1_velocityY));
            normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_angularProjector1, body1_angularVelocity));

            normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector2X, body2_velocityX));
            normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector2Y, body2_velocityY));
            normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_angularProjector2, body2_angularVelocity));

            Vf normalDeltaImpulse = _mm256_mul_ps(normaldV, j_normalLimiter_compInvMass);

            normalDeltaImpulse = _mm256_max_ps(normalDeltaImpulse, _mm256_xor_ps(sign, j_normalLimiter_accumulatedImpulse));

            body1_velocityX = _mm256_add_ps(body1_velocityX, _mm256_mul_ps(j_normalLimiter_compMass1_linearX, normalDeltaImpulse));
            body1_velocityY = _mm256_add_ps(body1_velocityY, _mm256_mul_ps(j_normalLimiter_compMass1_linearY, normalDeltaImpulse));
            body1_angularVelocity = _mm256_add_ps(body1_angularVelocity, _mm256_mul_ps(j_normalLimiter_compMass1_angular, normalDeltaImpulse));

            body2_velocityX = _mm256_add_ps(body2_velocityX, _mm256_mul_ps(j_normalLimiter_compMass2_linearX, normalDeltaImpulse));
            body2_velocityY = _mm256_add_ps(body2_velocityY, _mm256_mul_ps(j_normalLimiter_compMass2_linearY, normalDeltaImpulse));
            body2_angularVelocity = _mm256_add_ps(body2_angularVelocity, _mm256_mul_ps(j_normalLimiter_compMass2_angular, normalDeltaImpulse));

            j_normalLimiter_accumulatedImpulse = _mm256_add_ps(j_normalLimiter_accumulatedImpulse, normalDeltaImpulse);

            Vf frictiondV = zero;

            frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_normalProjector1X, body1_velocityX));
            frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_normalProjector1Y, body1_velocityY));
            frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_angularProjector1, body1_angularVelocity));

            frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_normalProjector2X, body2_velocityX));
            frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_normalProjector2Y, body2_velocityY));
            frictiondV = _mm256_sub_ps(frictiondV, _mm256_mul_ps(j_frictionLimiter_angularProjector2, body2_angularVelocity));

            Vf frictionDeltaImpulse = _mm256_mul_ps(frictiondV, j_frictionLimiter_compInvMass);

            Vf reactionForce = j_normalLimiter_accumulatedImpulse;
            Vf accumulatedImpulse = j_frictionLimiter_accumulatedImpulse;

            Vf frictionForce = _mm256_add_ps(accumulatedImpulse, frictionDeltaImpulse);
            Vf reactionForceScaled = _mm256_mul_ps(reactionForce, _mm256_set1_ps(0.3f));

            Vf frictionForceAbs = _mm256_andnot_ps(sign, frictionForce);
            Vf reactionForceScaledSigned = _mm256_xor_ps(_mm256_and_ps(frictionForce, sign), reactionForceScaled);
            Vf frictionDeltaImpulseAdjusted = _mm256_sub_ps(reactionForceScaledSigned, accumulatedImpulse);

            Vf frictionSelector = _mm256_cmp_ps(frictionForceAbs, reactionForceScaled, _CMP_GT_OQ);

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

            Vf cumulativeImpulse = _mm256_max_ps(_mm256_andnot_ps(sign, normalDeltaImpulse), _mm256_andnot_ps(sign, frictionDeltaImpulse));

            Vf productive = _mm256_cmp_ps(cumulativeImpulse, _mm256_set1_ps(kProductiveImpulse), _CMP_GT_OQ);

            body1_lastIteration = _mm256_blendv_epi8(body1_lastIteration, iterationIndex0, productive);
            body2_lastIteration = _mm256_blendv_epi8(body2_lastIteration, iterationIndex0, productive);

            // this is a bit painful :(
            static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Storing assumes fixed layout");

            row0 = body1_velocityX;
            row1 = body1_velocityY;
            row2 = body1_angularVelocity;
            row3 = body1_lastIteration;

            row4 = body2_velocityX;
            row5 = body2_velocityY;
            row6 = body2_angularVelocity;
            row7 = body2_lastIteration;

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 0]].velocity.x, _mm256_extractf128_ps(row0, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 0]].velocity.x, _mm256_extractf128_ps(row0, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 1]].velocity.x, _mm256_extractf128_ps(row1, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 1]].velocity.x, _mm256_extractf128_ps(row1, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 2]].velocity.x, _mm256_extractf128_ps(row2, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 2]].velocity.x, _mm256_extractf128_ps(row2, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 3]].velocity.x, _mm256_extractf128_ps(row3, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 3]].velocity.x, _mm256_extractf128_ps(row3, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 4]].velocity.x, _mm256_extractf128_ps(row4, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 4]].velocity.x, _mm256_extractf128_ps(row4, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 5]].velocity.x, _mm256_extractf128_ps(row5, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 5]].velocity.x, _mm256_extractf128_ps(row5, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 6]].velocity.x, _mm256_extractf128_ps(row6, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 6]].velocity.x, _mm256_extractf128_ps(row6, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 7]].velocity.x, _mm256_extractf128_ps(row7, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 7]].velocity.x, _mm256_extractf128_ps(row7, 1));
        }
    }
#endif

#if defined(__AVX2__) && defined(__FMA__)
    NOINLINE void SolveJointsImpulsesSoAPacked_FMA(ContactJointPacked<16>* joint_packed, int jointStart, int jointCount, int iterationIndex)
    {
        typedef __m256 Vf;
        typedef __m256i Vi;

        assert(jointStart % 16 == 0 && jointCount % 16 == 0);

        Vf sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

        Vi iterationIndex0 = _mm256_set1_epi32(iterationIndex);
        Vi iterationIndex2 = _mm256_set1_epi32(iterationIndex - 2);

        for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 16)
        {
            int i = jointIndex;

            ContactJointPacked<16>& jointP = joint_packed[i >> 4];
            int iP_0 = 0;
            int iP_1 = 8;

            Vf zero = _mm256_setzero_ps();

            Vf row0, row1, row2, row3, row4, row5, row6, row7;

            static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Loading assumes fixed layout");

            row0 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 0]].velocity.x, &solveBodies[jointP.body2Index[iP_0 + 0]].velocity.x);
            row1 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 1]].velocity.x, &solveBodies[jointP.body2Index[iP_0 + 1]].velocity.x);
            row2 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 2]].velocity.x, &solveBodies[jointP.body2Index[iP_0 + 2]].velocity.x);
            row3 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 3]].velocity.x, &solveBodies[jointP.body2Index[iP_0 + 3]].velocity.x);
            row4 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 4]].velocity.x, &solveBodies[jointP.body2Index[iP_0 + 4]].velocity.x);
            row5 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 5]].velocity.x, &solveBodies[jointP.body2Index[iP_0 + 5]].velocity.x);
            row6 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 6]].velocity.x, &solveBodies[jointP.body2Index[iP_0 + 6]].velocity.x);
            row7 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 7]].velocity.x, &solveBodies[jointP.body2Index[iP_0 + 7]].velocity.x);

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            Vf body1_velocityX_0 = row0;
            Vf body1_velocityY_0 = row1;
            Vf body1_angularVelocity_0 = row2;
            Vi body1_lastIteration_0 = row3;

            Vf body2_velocityX_0 = row4;
            Vf body2_velocityY_0 = row5;
            Vf body2_angularVelocity_0 = row6;
            Vf body2_lastIteration_0 = row7;

            row0 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 0]].velocity.x, &solveBodies[jointP.body2Index[iP_1 + 0]].velocity.x);
            row1 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 1]].velocity.x, &solveBodies[jointP.body2Index[iP_1 + 1]].velocity.x);
            row2 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 2]].velocity.x, &solveBodies[jointP.body2Index[iP_1 + 2]].velocity.x);
            row3 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 3]].velocity.x, &solveBodies[jointP.body2Index[iP_1 + 3]].velocity.x);
            row4 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 4]].velocity.x, &solveBodies[jointP.body2Index[iP_1 + 4]].velocity.x);
            row5 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 5]].velocity.x, &solveBodies[jointP.body2Index[iP_1 + 5]].velocity.x);
            row6 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 6]].velocity.x, &solveBodies[jointP.body2Index[iP_1 + 6]].velocity.x);
            row7 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 7]].velocity.x, &solveBodies[jointP.body2Index[iP_1 + 7]].velocity.x);

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            Vf body1_velocityX_1 = row0;
            Vf body1_velocityY_1 = row1;
            Vf body1_angularVelocity_1 = row2;
            Vi body1_lastIteration_1 = row3;

            Vf body2_velocityX_1 = row4;
            Vf body2_velocityY_1 = row5;
            Vf body2_angularVelocity_1 = row6;
            Vi body2_lastIteration_1 = row7;

            Vi body_lastIteration_0 = _mm256_max_epi32(body1_lastIteration_0, body2_lastIteration_0);
            Vi body_lastIteration_1 = _mm256_max_epi32(body1_lastIteration_1, body2_lastIteration_1);

            Vi body_productive_0 = _mm256_cmpgt_epi32(body_lastIteration_0, iterationIndex2);
            Vi body_productive_1 = _mm256_cmpgt_epi32(body_lastIteration_1, iterationIndex2);
            Vi body_productive = _mm256_or_si256(body_productive_0, body_productive_1);

            if (_mm256_movemask_epi8(body_productive) == 0)
                continue;

            Vf j_normalLimiter_normalProjector1X_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1X[iP_0]);
            Vf j_normalLimiter_normalProjector1Y_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1Y[iP_0]);
            Vf j_normalLimiter_normalProjector2X_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2X[iP_0]);
            Vf j_normalLimiter_normalProjector2Y_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2Y[iP_0]);
            Vf j_normalLimiter_angularProjector1_0 = _mm256_load_ps(&jointP.normalLimiter_angularProjector1[iP_0]);
            Vf j_normalLimiter_angularProjector2_0 = _mm256_load_ps(&jointP.normalLimiter_angularProjector2[iP_0]);

            Vf j_normalLimiter_compMass1_linearX_0 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearX[iP_0]);
            Vf j_normalLimiter_compMass1_linearY_0 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearY[iP_0]);
            Vf j_normalLimiter_compMass2_linearX_0 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearX[iP_0]);
            Vf j_normalLimiter_compMass2_linearY_0 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearY[iP_0]);
            Vf j_normalLimiter_compMass1_angular_0 = _mm256_load_ps(&jointP.normalLimiter_compMass1_angular[iP_0]);
            Vf j_normalLimiter_compMass2_angular_0 = _mm256_load_ps(&jointP.normalLimiter_compMass2_angular[iP_0]);
            Vf j_normalLimiter_compInvMass_0 = _mm256_load_ps(&jointP.normalLimiter_compInvMass[iP_0]);
            Vf j_normalLimiter_accumulatedImpulse_0 = _mm256_load_ps(&jointP.normalLimiter_accumulatedImpulse[iP_0]);
            Vf j_normalLimiter_dstVelocity_0 = _mm256_load_ps(&jointP.normalLimiter_dstVelocity[iP_0]);

            Vf j_frictionLimiter_normalProjector1X_0 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector1X[iP_0]);
            Vf j_frictionLimiter_normalProjector1Y_0 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector1Y[iP_0]);
            Vf j_frictionLimiter_normalProjector2X_0 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector2X[iP_0]);
            Vf j_frictionLimiter_normalProjector2Y_0 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector2Y[iP_0]);
            Vf j_frictionLimiter_angularProjector1_0 = _mm256_load_ps(&jointP.frictionLimiter_angularProjector1[iP_0]);
            Vf j_frictionLimiter_angularProjector2_0 = _mm256_load_ps(&jointP.frictionLimiter_angularProjector2[iP_0]);

            Vf j_frictionLimiter_compMass1_linearX_0 = _mm256_load_ps(&jointP.frictionLimiter_compMass1_linearX[iP_0]);
            Vf j_frictionLimiter_compMass1_linearY_0 = _mm256_load_ps(&jointP.frictionLimiter_compMass1_linearY[iP_0]);
            Vf j_frictionLimiter_compMass2_linearX_0 = _mm256_load_ps(&jointP.frictionLimiter_compMass2_linearX[iP_0]);
            Vf j_frictionLimiter_compMass2_linearY_0 = _mm256_load_ps(&jointP.frictionLimiter_compMass2_linearY[iP_0]);
            Vf j_frictionLimiter_compMass1_angular_0 = _mm256_load_ps(&jointP.frictionLimiter_compMass1_angular[iP_0]);
            Vf j_frictionLimiter_compMass2_angular_0 = _mm256_load_ps(&jointP.frictionLimiter_compMass2_angular[iP_0]);
            Vf j_frictionLimiter_compInvMass_0 = _mm256_load_ps(&jointP.frictionLimiter_compInvMass[iP_0]);
            Vf j_frictionLimiter_accumulatedImpulse_0 = _mm256_load_ps(&jointP.frictionLimiter_accumulatedImpulse[iP_0]);

            Vf j_normalLimiter_normalProjector1X_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1X[iP_1]);
            Vf j_normalLimiter_normalProjector1Y_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1Y[iP_1]);
            Vf j_normalLimiter_normalProjector2X_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2X[iP_1]);
            Vf j_normalLimiter_normalProjector2Y_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2Y[iP_1]);
            Vf j_normalLimiter_angularProjector1_1 = _mm256_load_ps(&jointP.normalLimiter_angularProjector1[iP_1]);
            Vf j_normalLimiter_angularProjector2_1 = _mm256_load_ps(&jointP.normalLimiter_angularProjector2[iP_1]);

            Vf j_normalLimiter_compMass1_linearX_1 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearX[iP_1]);
            Vf j_normalLimiter_compMass1_linearY_1 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearY[iP_1]);
            Vf j_normalLimiter_compMass2_linearX_1 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearX[iP_1]);
            Vf j_normalLimiter_compMass2_linearY_1 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearY[iP_1]);
            Vf j_normalLimiter_compMass1_angular_1 = _mm256_load_ps(&jointP.normalLimiter_compMass1_angular[iP_1]);
            Vf j_normalLimiter_compMass2_angular_1 = _mm256_load_ps(&jointP.normalLimiter_compMass2_angular[iP_1]);
            Vf j_normalLimiter_compInvMass_1 = _mm256_load_ps(&jointP.normalLimiter_compInvMass[iP_1]);
            Vf j_normalLimiter_accumulatedImpulse_1 = _mm256_load_ps(&jointP.normalLimiter_accumulatedImpulse[iP_1]);
            Vf j_normalLimiter_dstVelocity_1 = _mm256_load_ps(&jointP.normalLimiter_dstVelocity[iP_1]);

            Vf j_frictionLimiter_normalProjector1X_1 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector1X[iP_1]);
            Vf j_frictionLimiter_normalProjector1Y_1 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector1Y[iP_1]);
            Vf j_frictionLimiter_normalProjector2X_1 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector2X[iP_1]);
            Vf j_frictionLimiter_normalProjector2Y_1 = _mm256_load_ps(&jointP.frictionLimiter_normalProjector2Y[iP_1]);
            Vf j_frictionLimiter_angularProjector1_1 = _mm256_load_ps(&jointP.frictionLimiter_angularProjector1[iP_1]);
            Vf j_frictionLimiter_angularProjector2_1 = _mm256_load_ps(&jointP.frictionLimiter_angularProjector2[iP_1]);

            Vf j_frictionLimiter_compMass1_linearX_1 = _mm256_load_ps(&jointP.frictionLimiter_compMass1_linearX[iP_1]);
            Vf j_frictionLimiter_compMass1_linearY_1 = _mm256_load_ps(&jointP.frictionLimiter_compMass1_linearY[iP_1]);
            Vf j_frictionLimiter_compMass2_linearX_1 = _mm256_load_ps(&jointP.frictionLimiter_compMass2_linearX[iP_1]);
            Vf j_frictionLimiter_compMass2_linearY_1 = _mm256_load_ps(&jointP.frictionLimiter_compMass2_linearY[iP_1]);
            Vf j_frictionLimiter_compMass1_angular_1 = _mm256_load_ps(&jointP.frictionLimiter_compMass1_angular[iP_1]);
            Vf j_frictionLimiter_compMass2_angular_1 = _mm256_load_ps(&jointP.frictionLimiter_compMass2_angular[iP_1]);
            Vf j_frictionLimiter_compInvMass_1 = _mm256_load_ps(&jointP.frictionLimiter_compInvMass[iP_1]);
            Vf j_frictionLimiter_accumulatedImpulse_1 = _mm256_load_ps(&jointP.frictionLimiter_accumulatedImpulse[iP_1]);

            Vf normaldV1_0 = j_normalLimiter_dstVelocity_0;

            normaldV1_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1X_0, body1_velocityX_0, normaldV1_0);
            normaldV1_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1Y_0, body1_velocityY_0, normaldV1_0);
            normaldV1_0 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector1_0, body1_angularVelocity_0, normaldV1_0);

            Vf normaldV2_0 = zero;

            normaldV2_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2X_0, body2_velocityX_0, normaldV2_0);
            normaldV2_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2Y_0, body2_velocityY_0, normaldV2_0);
            normaldV2_0 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector2_0, body2_angularVelocity_0, normaldV2_0);

            Vf normaldV_0 = _mm256_add_ps(normaldV1_0, normaldV2_0);

            Vf normalDeltaImpulse_0 = _mm256_mul_ps(normaldV_0, j_normalLimiter_compInvMass_0);

            normalDeltaImpulse_0 = _mm256_max_ps(normalDeltaImpulse_0, _mm256_xor_ps(sign, j_normalLimiter_accumulatedImpulse_0));

            body1_velocityX_0 = _mm256_fmadd_ps(j_normalLimiter_compMass1_linearX_0, normalDeltaImpulse_0, body1_velocityX_0);
            body1_velocityY_0 = _mm256_fmadd_ps(j_normalLimiter_compMass1_linearY_0, normalDeltaImpulse_0, body1_velocityY_0);
            body1_angularVelocity_0 = _mm256_fmadd_ps(j_normalLimiter_compMass1_angular_0, normalDeltaImpulse_0, body1_angularVelocity_0);

            body2_velocityX_0 = _mm256_fmadd_ps(j_normalLimiter_compMass2_linearX_0, normalDeltaImpulse_0, body2_velocityX_0);
            body2_velocityY_0 = _mm256_fmadd_ps(j_normalLimiter_compMass2_linearY_0, normalDeltaImpulse_0, body2_velocityY_0);
            body2_angularVelocity_0 = _mm256_fmadd_ps(j_normalLimiter_compMass2_angular_0, normalDeltaImpulse_0, body2_angularVelocity_0);

            j_normalLimiter_accumulatedImpulse_0 = _mm256_add_ps(j_normalLimiter_accumulatedImpulse_0, normalDeltaImpulse_0);

            Vf frictiondV0_0 = zero;

            frictiondV0_0 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector1X_0, body1_velocityX_0, frictiondV0_0);
            frictiondV0_0 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector1Y_0, body1_velocityY_0, frictiondV0_0);
            frictiondV0_0 = _mm256_fnmadd_ps(j_frictionLimiter_angularProjector1_0, body1_angularVelocity_0, frictiondV0_0);

            Vf frictiondV1_0 = zero;

            frictiondV1_0 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector2X_0, body2_velocityX_0, frictiondV1_0);
            frictiondV1_0 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector2Y_0, body2_velocityY_0, frictiondV1_0);
            frictiondV1_0 = _mm256_fnmadd_ps(j_frictionLimiter_angularProjector2_0, body2_angularVelocity_0, frictiondV1_0);

            Vf frictiondV_0 = _mm256_add_ps(frictiondV0_0, frictiondV1_0);

            Vf frictionDeltaImpulse_0 = _mm256_mul_ps(frictiondV_0, j_frictionLimiter_compInvMass_0);

            Vf reactionForce_0 = j_normalLimiter_accumulatedImpulse_0;
            Vf accumulatedImpulse_0 = j_frictionLimiter_accumulatedImpulse_0;

            Vf frictionForce_0 = _mm256_add_ps(accumulatedImpulse_0, frictionDeltaImpulse_0);
            Vf reactionForceScaled_0 = _mm256_mul_ps(reactionForce_0, _mm256_set1_ps(0.3f));

            Vf frictionForceAbs_0 = _mm256_andnot_ps(sign, frictionForce_0);
            Vf reactionForceScaledSigned_0 = _mm256_xor_ps(_mm256_and_ps(frictionForce_0, sign), reactionForceScaled_0);
            Vf frictionDeltaImpulseAdjusted_0 = _mm256_sub_ps(reactionForceScaledSigned_0, accumulatedImpulse_0);

            Vf frictionSelector_0 = _mm256_cmp_ps(frictionForceAbs_0, reactionForceScaled_0, _CMP_GT_OQ);

            frictionDeltaImpulse_0 = _mm256_blendv_ps(frictionDeltaImpulse_0, frictionDeltaImpulseAdjusted_0, frictionSelector_0);

            j_frictionLimiter_accumulatedImpulse_0 = _mm256_add_ps(j_frictionLimiter_accumulatedImpulse_0, frictionDeltaImpulse_0);

            body1_velocityX_0 = _mm256_fmadd_ps(j_frictionLimiter_compMass1_linearX_0, frictionDeltaImpulse_0, body1_velocityX_0);
            body1_velocityY_0 = _mm256_fmadd_ps(j_frictionLimiter_compMass1_linearY_0, frictionDeltaImpulse_0, body1_velocityY_0);
            body1_angularVelocity_0 = _mm256_fmadd_ps(j_frictionLimiter_compMass1_angular_0, frictionDeltaImpulse_0, body1_angularVelocity_0);

            body2_velocityX_0 = _mm256_fmadd_ps(j_frictionLimiter_compMass2_linearX_0, frictionDeltaImpulse_0, body2_velocityX_0);
            body2_velocityY_0 = _mm256_fmadd_ps(j_frictionLimiter_compMass2_linearY_0, frictionDeltaImpulse_0, body2_velocityY_0);
            body2_angularVelocity_0 = _mm256_fmadd_ps(j_frictionLimiter_compMass2_angular_0, frictionDeltaImpulse_0, body2_angularVelocity_0);

            Vf normaldV1_1 = j_normalLimiter_dstVelocity_1;

            normaldV1_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1X_1, body1_velocityX_1, normaldV1_1);
            normaldV1_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1Y_1, body1_velocityY_1, normaldV1_1);
            normaldV1_1 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector1_1, body1_angularVelocity_1, normaldV1_1);

            Vf normaldV2_1 = zero;

            normaldV2_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2X_1, body2_velocityX_1, normaldV2_1);
            normaldV2_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2Y_1, body2_velocityY_1, normaldV2_1);
            normaldV2_1 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector2_1, body2_angularVelocity_1, normaldV2_1);

            Vf normaldV_1 = _mm256_add_ps(normaldV1_1, normaldV2_1);

            Vf normalDeltaImpulse_1 = _mm256_mul_ps(normaldV_1, j_normalLimiter_compInvMass_1);

            normalDeltaImpulse_1 = _mm256_max_ps(normalDeltaImpulse_1, _mm256_xor_ps(sign, j_normalLimiter_accumulatedImpulse_1));

            body1_velocityX_1 = _mm256_fmadd_ps(j_normalLimiter_compMass1_linearX_1, normalDeltaImpulse_1, body1_velocityX_1);
            body1_velocityY_1 = _mm256_fmadd_ps(j_normalLimiter_compMass1_linearY_1, normalDeltaImpulse_1, body1_velocityY_1);
            body1_angularVelocity_1 = _mm256_fmadd_ps(j_normalLimiter_compMass1_angular_1, normalDeltaImpulse_1, body1_angularVelocity_1);

            body2_velocityX_1 = _mm256_fmadd_ps(j_normalLimiter_compMass2_linearX_1, normalDeltaImpulse_1, body2_velocityX_1);
            body2_velocityY_1 = _mm256_fmadd_ps(j_normalLimiter_compMass2_linearY_1, normalDeltaImpulse_1, body2_velocityY_1);
            body2_angularVelocity_1 = _mm256_fmadd_ps(j_normalLimiter_compMass2_angular_1, normalDeltaImpulse_1, body2_angularVelocity_1);

            j_normalLimiter_accumulatedImpulse_1 = _mm256_add_ps(j_normalLimiter_accumulatedImpulse_1, normalDeltaImpulse_1);

            Vf frictiondV0_1 = zero;

            frictiondV0_1 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector1X_1, body1_velocityX_1, frictiondV0_1);
            frictiondV0_1 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector1Y_1, body1_velocityY_1, frictiondV0_1);
            frictiondV0_1 = _mm256_fnmadd_ps(j_frictionLimiter_angularProjector1_1, body1_angularVelocity_1, frictiondV0_1);

            Vf frictiondV1_1 = zero;

            frictiondV1_1 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector2X_1, body2_velocityX_1, frictiondV1_1);
            frictiondV1_1 = _mm256_fnmadd_ps(j_frictionLimiter_normalProjector2Y_1, body2_velocityY_1, frictiondV1_1);
            frictiondV1_1 = _mm256_fnmadd_ps(j_frictionLimiter_angularProjector2_1, body2_angularVelocity_1, frictiondV1_1);

            Vf frictiondV_1 = _mm256_add_ps(frictiondV0_1, frictiondV1_1);

            Vf frictionDeltaImpulse_1 = _mm256_mul_ps(frictiondV_1, j_frictionLimiter_compInvMass_1);

            Vf reactionForce_1 = j_normalLimiter_accumulatedImpulse_1;
            Vf accumulatedImpulse_1 = j_frictionLimiter_accumulatedImpulse_1;

            Vf frictionForce_1 = _mm256_add_ps(accumulatedImpulse_1, frictionDeltaImpulse_1);
            Vf reactionForceScaled_1 = _mm256_mul_ps(reactionForce_1, _mm256_set1_ps(0.3f));

            Vf frictionForceAbs_1 = _mm256_andnot_ps(sign, frictionForce_1);
            Vf reactionForceScaledSigned_1 = _mm256_xor_ps(_mm256_and_ps(frictionForce_1, sign), reactionForceScaled_1);
            Vf frictionDeltaImpulseAdjusted_1 = _mm256_sub_ps(reactionForceScaledSigned_1, accumulatedImpulse_1);

            Vf frictionSelector_1 = _mm256_cmp_ps(frictionForceAbs_1, reactionForceScaled_1, _CMP_GT_OQ);

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

            Vf cumulativeImpulse_0 = _mm256_max_ps(_mm256_andnot_ps(sign, normalDeltaImpulse_0), _mm256_andnot_ps(sign, frictionDeltaImpulse_0));
            Vf cumulativeImpulse_1 = _mm256_max_ps(_mm256_andnot_ps(sign, normalDeltaImpulse_1), _mm256_andnot_ps(sign, frictionDeltaImpulse_1));

            Vf productive_0 = _mm256_cmp_ps(cumulativeImpulse_0, _mm256_set1_ps(kProductiveImpulse), _CMP_GT_OQ);
            Vf productive_1 = _mm256_cmp_ps(cumulativeImpulse_1, _mm256_set1_ps(kProductiveImpulse), _CMP_GT_OQ);

            body1_lastIteration_0 = _mm256_blendv_epi8(body1_lastIteration_0, iterationIndex0, productive_0);
            body2_lastIteration_0 = _mm256_blendv_epi8(body2_lastIteration_0, iterationIndex0, productive_0);

            body1_lastIteration_1 = _mm256_blendv_epi8(body1_lastIteration_1, iterationIndex0, productive_1);
            body2_lastIteration_1 = _mm256_blendv_epi8(body2_lastIteration_1, iterationIndex0, productive_1);

            // this is a bit painful :(
            static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Storing assumes fixed layout");

            row0 = body1_velocityX_0;
            row1 = body1_velocityY_0;
            row2 = body1_angularVelocity_0;
            row3 = body1_lastIteration_0;

            row4 = body2_velocityX_0;
            row5 = body2_velocityY_0;
            row6 = body2_angularVelocity_0;
            row7 = body2_lastIteration_0;

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 0]].velocity.x, _mm256_extractf128_ps(row0, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 0]].velocity.x, _mm256_extractf128_ps(row0, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 1]].velocity.x, _mm256_extractf128_ps(row1, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 1]].velocity.x, _mm256_extractf128_ps(row1, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 2]].velocity.x, _mm256_extractf128_ps(row2, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 2]].velocity.x, _mm256_extractf128_ps(row2, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 3]].velocity.x, _mm256_extractf128_ps(row3, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 3]].velocity.x, _mm256_extractf128_ps(row3, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 4]].velocity.x, _mm256_extractf128_ps(row4, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 4]].velocity.x, _mm256_extractf128_ps(row4, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 5]].velocity.x, _mm256_extractf128_ps(row5, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 5]].velocity.x, _mm256_extractf128_ps(row5, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 6]].velocity.x, _mm256_extractf128_ps(row6, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 6]].velocity.x, _mm256_extractf128_ps(row6, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 7]].velocity.x, _mm256_extractf128_ps(row7, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 7]].velocity.x, _mm256_extractf128_ps(row7, 1));

            row0 = body1_velocityX_1;
            row1 = body1_velocityY_1;
            row2 = body1_angularVelocity_1;
            row3 = body1_lastIteration_1;

            row4 = body2_velocityX_1;
            row5 = body2_velocityY_1;
            row6 = body2_angularVelocity_1;
            row7 = body2_lastIteration_1;

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 0]].velocity.x, _mm256_extractf128_ps(row0, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 0]].velocity.x, _mm256_extractf128_ps(row0, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 1]].velocity.x, _mm256_extractf128_ps(row1, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 1]].velocity.x, _mm256_extractf128_ps(row1, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 2]].velocity.x, _mm256_extractf128_ps(row2, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 2]].velocity.x, _mm256_extractf128_ps(row2, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 3]].velocity.x, _mm256_extractf128_ps(row3, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 3]].velocity.x, _mm256_extractf128_ps(row3, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 4]].velocity.x, _mm256_extractf128_ps(row4, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 4]].velocity.x, _mm256_extractf128_ps(row4, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 5]].velocity.x, _mm256_extractf128_ps(row5, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 5]].velocity.x, _mm256_extractf128_ps(row5, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 6]].velocity.x, _mm256_extractf128_ps(row6, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 6]].velocity.x, _mm256_extractf128_ps(row6, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 7]].velocity.x, _mm256_extractf128_ps(row7, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 7]].velocity.x, _mm256_extractf128_ps(row7, 1));
        }
    }

#endif

    template <int N>
    NOINLINE void SolveJointsDisplacementSoAPacked(ContactJointPacked<N>* joint_packed, int jointStart, int jointCount, int iterationIndex)
    {
        for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex++)
        {
            int i = jointIndex;

            ContactJointPacked<N>& jointP = joint_packed[unsigned(i) / N];
            int iP = i & (N - 1);

            SolveBody* body1 = &solveBodies[jointP.body1Index[iP]];
            SolveBody* body2 = &solveBodies[jointP.body2Index[iP]];

            if (body1->lastDisplacementIteration < iterationIndex - 1 && body2->lastDisplacementIteration < iterationIndex - 1)
                continue;

            float dV = jointP.normalLimiter_dstDisplacingVelocity[iP];

            dV -= jointP.normalLimiter_normalProjector1X[iP] * body1->displacingVelocity.x;
            dV -= jointP.normalLimiter_normalProjector1Y[iP] * body1->displacingVelocity.y;
            dV -= jointP.normalLimiter_angularProjector1[iP] * body1->displacingAngularVelocity;

            dV -= jointP.normalLimiter_normalProjector2X[iP] * body2->displacingVelocity.x;
            dV -= jointP.normalLimiter_normalProjector2Y[iP] * body2->displacingVelocity.y;
            dV -= jointP.normalLimiter_angularProjector2[iP] * body2->displacingAngularVelocity;

            float displacingDeltaImpulse = dV * jointP.normalLimiter_compInvMass[iP];

            if (displacingDeltaImpulse + jointP.normalLimiter_accumulatedDisplacingImpulse[iP] < 0.0f)
                displacingDeltaImpulse = -jointP.normalLimiter_accumulatedDisplacingImpulse[iP];

            body1->displacingVelocity.x += jointP.normalLimiter_compMass1_linearX[iP] * displacingDeltaImpulse;
            body1->displacingVelocity.y += jointP.normalLimiter_compMass1_linearY[iP] * displacingDeltaImpulse;
            body1->displacingAngularVelocity += jointP.normalLimiter_compMass1_angular[iP] * displacingDeltaImpulse;

            body2->displacingVelocity.x += jointP.normalLimiter_compMass2_linearX[iP] * displacingDeltaImpulse;
            body2->displacingVelocity.y += jointP.normalLimiter_compMass2_linearY[iP] * displacingDeltaImpulse;
            body2->displacingAngularVelocity += jointP.normalLimiter_compMass2_angular[iP] * displacingDeltaImpulse;

            jointP.normalLimiter_accumulatedDisplacingImpulse[iP] += displacingDeltaImpulse;

            if (fabsf(displacingDeltaImpulse) > kProductiveImpulse)
            {
                body1->lastDisplacementIteration = iterationIndex;
                body2->lastDisplacementIteration = iterationIndex;
            }
        }
    }

    NOINLINE void SolveJointsDisplacementSoAPacked_SSE2(ContactJointPacked<4>* joint_packed, int jointStart, int jointCount, int iterationIndex)
    {
        typedef __m128 Vf;
        typedef __m128i Vi;

        assert(jointStart % 4 == 0 && jointCount % 4 == 0);

        Vf sign = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));

        Vi iterationIndex0 = _mm_set1_epi32(iterationIndex);
        Vi iterationIndex2 = _mm_set1_epi32(iterationIndex - 2);

        for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 4)
        {
            int i = jointIndex;

            ContactJointPacked<4>& jointP = joint_packed[i >> 3];
            int iP = 0;

            Vf zero = _mm_setzero_ps();

            __m128 row0, row1, row2, row3;

            static_assert(offsetof(SolveBody, displacingVelocity) == 16 && offsetof(SolveBody, displacingAngularVelocity) == 24, "Loading assumes fixed layout");

            row0 = _mm_load_ps(&solveBodies[jointP.body1Index[iP + 0]].displacingVelocity.x);
            row1 = _mm_load_ps(&solveBodies[jointP.body1Index[iP + 1]].displacingVelocity.x);
            row2 = _mm_load_ps(&solveBodies[jointP.body1Index[iP + 2]].displacingVelocity.x);
            row3 = _mm_load_ps(&solveBodies[jointP.body1Index[iP + 3]].displacingVelocity.x);

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            Vf body1_displacingVelocityX = row0;
            Vf body1_displacingVelocityY = row1;
            Vf body1_displacingAngularVelocity = row2;
            Vi body1_lastDisplacementIteration = row3;

            row0 = _mm_load_ps(&solveBodies[jointP.body2Index[iP + 0]].displacingVelocity.x);
            row1 = _mm_load_ps(&solveBodies[jointP.body2Index[iP + 1]].displacingVelocity.x);
            row2 = _mm_load_ps(&solveBodies[jointP.body2Index[iP + 2]].displacingVelocity.x);
            row3 = _mm_load_ps(&solveBodies[jointP.body2Index[iP + 3]].displacingVelocity.x);

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            Vf body2_displacingVelocityX = row0;
            Vf body2_displacingVelocityY = row1;
            Vf body2_displacingAngularVelocity = row2;
            Vi body2_lastDisplacementIteration = row3;

            Vi body1_productive = _mm_cmpgt_epi32(body1_lastDisplacementIteration, iterationIndex2);
            Vi body2_productive = _mm_cmpgt_epi32(body2_lastDisplacementIteration, iterationIndex2);
            Vi body_productive = _mm_or_si128(body1_productive, body2_productive);

            if (_mm_movemask_epi8(body_productive) == 0)
                continue;

            Vf j_normalLimiter_normalProjector1X = _mm_load_ps(&jointP.normalLimiter_normalProjector1X[iP]);
            Vf j_normalLimiter_normalProjector1Y = _mm_load_ps(&jointP.normalLimiter_normalProjector1Y[iP]);
            Vf j_normalLimiter_normalProjector2X = _mm_load_ps(&jointP.normalLimiter_normalProjector2X[iP]);
            Vf j_normalLimiter_normalProjector2Y = _mm_load_ps(&jointP.normalLimiter_normalProjector2Y[iP]);
            Vf j_normalLimiter_angularProjector1 = _mm_load_ps(&jointP.normalLimiter_angularProjector1[iP]);
            Vf j_normalLimiter_angularProjector2 = _mm_load_ps(&jointP.normalLimiter_angularProjector2[iP]);

            Vf j_normalLimiter_compMass1_linearX = _mm_load_ps(&jointP.normalLimiter_compMass1_linearX[iP]);
            Vf j_normalLimiter_compMass1_linearY = _mm_load_ps(&jointP.normalLimiter_compMass1_linearY[iP]);
            Vf j_normalLimiter_compMass2_linearX = _mm_load_ps(&jointP.normalLimiter_compMass2_linearX[iP]);
            Vf j_normalLimiter_compMass2_linearY = _mm_load_ps(&jointP.normalLimiter_compMass2_linearY[iP]);
            Vf j_normalLimiter_compMass1_angular = _mm_load_ps(&jointP.normalLimiter_compMass1_angular[iP]);
            Vf j_normalLimiter_compMass2_angular = _mm_load_ps(&jointP.normalLimiter_compMass2_angular[iP]);
            Vf j_normalLimiter_compInvMass = _mm_load_ps(&jointP.normalLimiter_compInvMass[iP]);
            Vf j_normalLimiter_dstDisplacingVelocity = _mm_load_ps(&jointP.normalLimiter_dstDisplacingVelocity[iP]);
            Vf j_normalLimiter_accumulatedDisplacingImpulse = _mm_load_ps(&jointP.normalLimiter_accumulatedDisplacingImpulse[iP]);

            Vf dV = j_normalLimiter_dstDisplacingVelocity;

            dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_normalProjector1X, body1_displacingVelocityX));
            dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_normalProjector1Y, body1_displacingVelocityY));
            dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_angularProjector1, body1_displacingAngularVelocity));

            dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_normalProjector2X, body2_displacingVelocityX));
            dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_normalProjector2Y, body2_displacingVelocityY));
            dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_angularProjector2, body2_displacingAngularVelocity));

            Vf displacingDeltaImpulse = _mm_mul_ps(dV, j_normalLimiter_compInvMass);

            displacingDeltaImpulse = _mm_max_ps(displacingDeltaImpulse, _mm_xor_ps(sign, j_normalLimiter_accumulatedDisplacingImpulse));

            body1_displacingVelocityX = _mm_add_ps(body1_displacingVelocityX, _mm_mul_ps(j_normalLimiter_compMass1_linearX, displacingDeltaImpulse));
            body1_displacingVelocityY = _mm_add_ps(body1_displacingVelocityY, _mm_mul_ps(j_normalLimiter_compMass1_linearY, displacingDeltaImpulse));
            body1_displacingAngularVelocity = _mm_add_ps(body1_displacingAngularVelocity, _mm_mul_ps(j_normalLimiter_compMass1_angular, displacingDeltaImpulse));

            body2_displacingVelocityX = _mm_add_ps(body2_displacingVelocityX, _mm_mul_ps(j_normalLimiter_compMass2_linearX, displacingDeltaImpulse));
            body2_displacingVelocityY = _mm_add_ps(body2_displacingVelocityY, _mm_mul_ps(j_normalLimiter_compMass2_linearY, displacingDeltaImpulse));
            body2_displacingAngularVelocity = _mm_add_ps(body2_displacingAngularVelocity, _mm_mul_ps(j_normalLimiter_compMass2_angular, displacingDeltaImpulse));

            j_normalLimiter_accumulatedDisplacingImpulse = _mm_add_ps(j_normalLimiter_accumulatedDisplacingImpulse, displacingDeltaImpulse);

            _mm_store_ps(&jointP.normalLimiter_accumulatedDisplacingImpulse[iP], j_normalLimiter_accumulatedDisplacingImpulse);

            Vf productive = _mm_cmpgt_ps(_mm_andnot_ps(sign, displacingDeltaImpulse), _mm_set1_ps(kProductiveImpulse));

            body1_lastDisplacementIteration = _mm_or_si128(_mm_andnot_si128(productive, body1_lastDisplacementIteration), _mm_and_si128(iterationIndex0, productive));
            body2_lastDisplacementIteration = _mm_or_si128(_mm_andnot_si128(productive, body2_lastDisplacementIteration), _mm_and_si128(iterationIndex0, productive));

            // this is a bit painful :(
            static_assert(offsetof(SolveBody, displacingVelocity) == 16 && offsetof(SolveBody, displacingAngularVelocity) == 24, "Storing assumes fixed layout");

            row0 = body1_displacingVelocityX;
            row1 = body1_displacingVelocityY;
            row2 = body1_displacingAngularVelocity;
            row3 = body1_lastDisplacementIteration;

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 0]].displacingVelocity.x, row0);
            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 1]].displacingVelocity.x, row1);
            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 2]].displacingVelocity.x, row2);
            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 3]].displacingVelocity.x, row3);

            row0 = body2_displacingVelocityX;
            row1 = body2_displacingVelocityY;
            row2 = body2_displacingAngularVelocity;
            row3 = body2_lastDisplacementIteration;

            _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 0]].displacingVelocity.x, row0);
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 1]].displacingVelocity.x, row1);
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 2]].displacingVelocity.x, row2);
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 3]].displacingVelocity.x, row3);
        }
    }

#ifdef __AVX2__
    NOINLINE void SolveJointsDisplacementSoAPacked_AVX2(ContactJointPacked<8>* joint_packed, int jointStart, int jointCount, int iterationIndex)
    {
        typedef __m256 Vf;
        typedef __m256i Vi;

        assert(jointStart % 8 == 0 && jointCount % 8 == 0);

        Vf sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

        Vi iterationIndex0 = _mm256_set1_epi32(iterationIndex);
        Vi iterationIndex2 = _mm256_set1_epi32(iterationIndex - 2);

        for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 8)
        {
            int i = jointIndex;

            ContactJointPacked<8>& jointP = joint_packed[i >> 3];
            int iP = 0;

            Vf zero = _mm256_setzero_ps();

            Vf row0, row1, row2, row3, row4, row5, row6, row7;

            static_assert(offsetof(SolveBody, displacingVelocity) == 16 && offsetof(SolveBody, displacingAngularVelocity) == 24, "Loading assumes fixed layout");

            row0 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 0]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP + 0]].displacingVelocity.x);
            row1 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 1]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP + 1]].displacingVelocity.x);
            row2 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 2]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP + 2]].displacingVelocity.x);
            row3 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 3]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP + 3]].displacingVelocity.x);
            row4 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 4]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP + 4]].displacingVelocity.x);
            row5 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 5]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP + 5]].displacingVelocity.x);
            row6 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 6]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP + 6]].displacingVelocity.x);
            row7 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP + 7]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP + 7]].displacingVelocity.x);

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            Vf body1_displacingVelocityX = row0;
            Vf body1_displacingVelocityY = row1;
            Vf body1_displacingAngularVelocity = row2;
            Vi body1_lastDisplacementIteration = row3;

            Vf body2_displacingVelocityX = row4;
            Vf body2_displacingVelocityY = row5;
            Vf body2_displacingAngularVelocity = row6;
            Vi body2_lastDisplacementIteration = row7;

            Vi body_lastDisplacementIteration = _mm256_max_epi32(body1_lastDisplacementIteration, body2_lastDisplacementIteration);
            Vi body_productive = _mm256_cmpgt_epi32(body_lastDisplacementIteration, iterationIndex2);

            if (_mm256_movemask_epi8(body_productive) == 0)
                continue;

            Vf j_normalLimiter_normalProjector1X = _mm256_load_ps(&jointP.normalLimiter_normalProjector1X[iP]);
            Vf j_normalLimiter_normalProjector1Y = _mm256_load_ps(&jointP.normalLimiter_normalProjector1Y[iP]);
            Vf j_normalLimiter_normalProjector2X = _mm256_load_ps(&jointP.normalLimiter_normalProjector2X[iP]);
            Vf j_normalLimiter_normalProjector2Y = _mm256_load_ps(&jointP.normalLimiter_normalProjector2Y[iP]);
            Vf j_normalLimiter_angularProjector1 = _mm256_load_ps(&jointP.normalLimiter_angularProjector1[iP]);
            Vf j_normalLimiter_angularProjector2 = _mm256_load_ps(&jointP.normalLimiter_angularProjector2[iP]);

            Vf j_normalLimiter_compMass1_linearX = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearX[iP]);
            Vf j_normalLimiter_compMass1_linearY = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearY[iP]);
            Vf j_normalLimiter_compMass2_linearX = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearX[iP]);
            Vf j_normalLimiter_compMass2_linearY = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearY[iP]);
            Vf j_normalLimiter_compMass1_angular = _mm256_load_ps(&jointP.normalLimiter_compMass1_angular[iP]);
            Vf j_normalLimiter_compMass2_angular = _mm256_load_ps(&jointP.normalLimiter_compMass2_angular[iP]);
            Vf j_normalLimiter_compInvMass = _mm256_load_ps(&jointP.normalLimiter_compInvMass[iP]);
            Vf j_normalLimiter_dstDisplacingVelocity = _mm256_load_ps(&jointP.normalLimiter_dstDisplacingVelocity[iP]);
            Vf j_normalLimiter_accumulatedDisplacingImpulse = _mm256_load_ps(&jointP.normalLimiter_accumulatedDisplacingImpulse[iP]);

            Vf dV = j_normalLimiter_dstDisplacingVelocity;

            dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector1X, body1_displacingVelocityX));
            dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector1Y, body1_displacingVelocityY));
            dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_angularProjector1, body1_displacingAngularVelocity));

            dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector2X, body2_displacingVelocityX));
            dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector2Y, body2_displacingVelocityY));
            dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_angularProjector2, body2_displacingAngularVelocity));

            Vf displacingDeltaImpulse = _mm256_mul_ps(dV, j_normalLimiter_compInvMass);

            displacingDeltaImpulse = _mm256_max_ps(displacingDeltaImpulse, _mm256_xor_ps(sign, j_normalLimiter_accumulatedDisplacingImpulse));

            body1_displacingVelocityX = _mm256_add_ps(body1_displacingVelocityX, _mm256_mul_ps(j_normalLimiter_compMass1_linearX, displacingDeltaImpulse));
            body1_displacingVelocityY = _mm256_add_ps(body1_displacingVelocityY, _mm256_mul_ps(j_normalLimiter_compMass1_linearY, displacingDeltaImpulse));
            body1_displacingAngularVelocity = _mm256_add_ps(body1_displacingAngularVelocity, _mm256_mul_ps(j_normalLimiter_compMass1_angular, displacingDeltaImpulse));

            body2_displacingVelocityX = _mm256_add_ps(body2_displacingVelocityX, _mm256_mul_ps(j_normalLimiter_compMass2_linearX, displacingDeltaImpulse));
            body2_displacingVelocityY = _mm256_add_ps(body2_displacingVelocityY, _mm256_mul_ps(j_normalLimiter_compMass2_linearY, displacingDeltaImpulse));
            body2_displacingAngularVelocity = _mm256_add_ps(body2_displacingAngularVelocity, _mm256_mul_ps(j_normalLimiter_compMass2_angular, displacingDeltaImpulse));

            j_normalLimiter_accumulatedDisplacingImpulse = _mm256_add_ps(j_normalLimiter_accumulatedDisplacingImpulse, displacingDeltaImpulse);

            _mm256_store_ps(&jointP.normalLimiter_accumulatedDisplacingImpulse[iP], j_normalLimiter_accumulatedDisplacingImpulse);

            Vf productive = _mm256_cmp_ps(_mm256_andnot_ps(sign, displacingDeltaImpulse), _mm256_set1_ps(kProductiveImpulse), _CMP_GT_OQ);

            body1_lastDisplacementIteration = _mm256_blendv_epi8(body1_lastDisplacementIteration, iterationIndex0, productive);
            body2_lastDisplacementIteration = _mm256_blendv_epi8(body2_lastDisplacementIteration, iterationIndex0, productive);

            // this is a bit painful :(
            static_assert(offsetof(SolveBody, displacingVelocity) == 16 && offsetof(SolveBody, displacingAngularVelocity) == 24, "Storing assumes fixed layout");

            row0 = body1_displacingVelocityX;
            row1 = body1_displacingVelocityY;
            row2 = body1_displacingAngularVelocity;
            row3 = body1_lastDisplacementIteration;

            row4 = body2_displacingVelocityX;
            row5 = body2_displacingVelocityY;
            row6 = body2_displacingAngularVelocity;
            row7 = body2_lastDisplacementIteration;

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 0]].displacingVelocity.x, _mm256_extractf128_ps(row0, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 0]].displacingVelocity.x, _mm256_extractf128_ps(row0, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 1]].displacingVelocity.x, _mm256_extractf128_ps(row1, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 1]].displacingVelocity.x, _mm256_extractf128_ps(row1, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 2]].displacingVelocity.x, _mm256_extractf128_ps(row2, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 2]].displacingVelocity.x, _mm256_extractf128_ps(row2, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 3]].displacingVelocity.x, _mm256_extractf128_ps(row3, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 3]].displacingVelocity.x, _mm256_extractf128_ps(row3, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 4]].displacingVelocity.x, _mm256_extractf128_ps(row4, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 4]].displacingVelocity.x, _mm256_extractf128_ps(row4, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 5]].displacingVelocity.x, _mm256_extractf128_ps(row5, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 5]].displacingVelocity.x, _mm256_extractf128_ps(row5, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 6]].displacingVelocity.x, _mm256_extractf128_ps(row6, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 6]].displacingVelocity.x, _mm256_extractf128_ps(row6, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP + 7]].displacingVelocity.x, _mm256_extractf128_ps(row7, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP + 7]].displacingVelocity.x, _mm256_extractf128_ps(row7, 1));
        }
    }
#endif

#if defined(__AVX2__) && defined(__FMA__)
    NOINLINE void SolveJointsDisplacementSoAPacked_FMA(ContactJointPacked<16>* joint_packed, int jointStart, int jointCount, int iterationIndex)
    {
        typedef __m256 Vf;
        typedef __m256i Vi;

        assert(jointStart % 16 == 0 && jointCount % 16 == 0);

        Vf sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

        Vi iterationIndex0 = _mm256_set1_epi32(iterationIndex);
        Vi iterationIndex2 = _mm256_set1_epi32(iterationIndex - 2);

        for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 16)
        {
            int i = jointIndex;

            ContactJointPacked<16>& jointP = joint_packed[i >> 4];
            int iP_0 = 0;
            int iP_1 = 8;

            Vf zero = _mm256_setzero_ps();

            Vf row0, row1, row2, row3, row4, row5, row6, row7;

            static_assert(offsetof(SolveBody, displacingVelocity) == 16 && offsetof(SolveBody, displacingAngularVelocity) == 24, "Loading assumes fixed layout");

            row0 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 0]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_0 + 0]].displacingVelocity.x);
            row1 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 1]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_0 + 1]].displacingVelocity.x);
            row2 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 2]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_0 + 2]].displacingVelocity.x);
            row3 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 3]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_0 + 3]].displacingVelocity.x);
            row4 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 4]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_0 + 4]].displacingVelocity.x);
            row5 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 5]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_0 + 5]].displacingVelocity.x);
            row6 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 6]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_0 + 6]].displacingVelocity.x);
            row7 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_0 + 7]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_0 + 7]].displacingVelocity.x);

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            Vf body1_displacingVelocityX_0 = row0;
            Vf body1_displacingVelocityY_0 = row1;
            Vf body1_displacingAngularVelocity_0 = row2;
            Vi body1_lastDisplacementIteration_0 = row3;

            Vf body2_displacingVelocityX_0 = row4;
            Vf body2_displacingVelocityY_0 = row5;
            Vf body2_displacingAngularVelocity_0 = row6;
            Vi body2_lastDisplacementIteration_0 = row7;

            row0 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 0]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_1 + 0]].displacingVelocity.x);
            row1 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 1]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_1 + 1]].displacingVelocity.x);
            row2 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 2]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_1 + 2]].displacingVelocity.x);
            row3 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 3]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_1 + 3]].displacingVelocity.x);
            row4 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 4]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_1 + 4]].displacingVelocity.x);
            row5 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 5]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_1 + 5]].displacingVelocity.x);
            row6 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 6]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_1 + 6]].displacingVelocity.x);
            row7 = _mm256_load2_m128(&solveBodies[jointP.body1Index[iP_1 + 7]].displacingVelocity.x, &solveBodies[jointP.body2Index[iP_1 + 7]].displacingVelocity.x);

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            Vf body1_displacingVelocityX_1 = row0;
            Vf body1_displacingVelocityY_1 = row1;
            Vf body1_displacingAngularVelocity_1 = row2;
            Vi body1_lastDisplacementIteration_1 = row3;

            Vf body2_displacingVelocityX_1 = row4;
            Vf body2_displacingVelocityY_1 = row5;
            Vf body2_displacingAngularVelocity_1 = row6;
            Vi body2_lastDisplacementIteration_1 = row7;

            Vi body_lastDisplacementIteration_0 = _mm256_max_epi32(body1_lastDisplacementIteration_0, body2_lastDisplacementIteration_0);
            Vi body_lastDisplacementIteration_1 = _mm256_max_epi32(body1_lastDisplacementIteration_1, body2_lastDisplacementIteration_1);

            Vi body_productive_0 = _mm256_cmpgt_epi32(body_lastDisplacementIteration_0, iterationIndex2);
            Vi body_productive_1 = _mm256_cmpgt_epi32(body_lastDisplacementIteration_1, iterationIndex2);
            Vi body_productive = _mm256_or_si256(body_productive_0, body_productive_1);

            if (_mm256_movemask_epi8(body_productive) == 0)
                continue;

            Vf j_normalLimiter_normalProjector1X_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1X[iP_0]);
            Vf j_normalLimiter_normalProjector1Y_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1Y[iP_0]);
            Vf j_normalLimiter_normalProjector2X_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2X[iP_0]);
            Vf j_normalLimiter_normalProjector2Y_0 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2Y[iP_0]);
            Vf j_normalLimiter_angularProjector1_0 = _mm256_load_ps(&jointP.normalLimiter_angularProjector1[iP_0]);
            Vf j_normalLimiter_angularProjector2_0 = _mm256_load_ps(&jointP.normalLimiter_angularProjector2[iP_0]);

            Vf j_normalLimiter_compMass1_linearX_0 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearX[iP_0]);
            Vf j_normalLimiter_compMass1_linearY_0 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearY[iP_0]);
            Vf j_normalLimiter_compMass2_linearX_0 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearX[iP_0]);
            Vf j_normalLimiter_compMass2_linearY_0 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearY[iP_0]);
            Vf j_normalLimiter_compMass1_angular_0 = _mm256_load_ps(&jointP.normalLimiter_compMass1_angular[iP_0]);
            Vf j_normalLimiter_compMass2_angular_0 = _mm256_load_ps(&jointP.normalLimiter_compMass2_angular[iP_0]);
            Vf j_normalLimiter_compInvMass_0 = _mm256_load_ps(&jointP.normalLimiter_compInvMass[iP_0]);
            Vf j_normalLimiter_dstDisplacingVelocity_0 = _mm256_load_ps(&jointP.normalLimiter_dstDisplacingVelocity[iP_0]);
            Vf j_normalLimiter_accumulatedDisplacingImpulse_0 = _mm256_load_ps(&jointP.normalLimiter_accumulatedDisplacingImpulse[iP_0]);

            Vf j_normalLimiter_normalProjector1X_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1X[iP_1]);
            Vf j_normalLimiter_normalProjector1Y_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector1Y[iP_1]);
            Vf j_normalLimiter_normalProjector2X_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2X[iP_1]);
            Vf j_normalLimiter_normalProjector2Y_1 = _mm256_load_ps(&jointP.normalLimiter_normalProjector2Y[iP_1]);
            Vf j_normalLimiter_angularProjector1_1 = _mm256_load_ps(&jointP.normalLimiter_angularProjector1[iP_1]);
            Vf j_normalLimiter_angularProjector2_1 = _mm256_load_ps(&jointP.normalLimiter_angularProjector2[iP_1]);

            Vf j_normalLimiter_compMass1_linearX_1 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearX[iP_1]);
            Vf j_normalLimiter_compMass1_linearY_1 = _mm256_load_ps(&jointP.normalLimiter_compMass1_linearY[iP_1]);
            Vf j_normalLimiter_compMass2_linearX_1 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearX[iP_1]);
            Vf j_normalLimiter_compMass2_linearY_1 = _mm256_load_ps(&jointP.normalLimiter_compMass2_linearY[iP_1]);
            Vf j_normalLimiter_compMass1_angular_1 = _mm256_load_ps(&jointP.normalLimiter_compMass1_angular[iP_1]);
            Vf j_normalLimiter_compMass2_angular_1 = _mm256_load_ps(&jointP.normalLimiter_compMass2_angular[iP_1]);
            Vf j_normalLimiter_compInvMass_1 = _mm256_load_ps(&jointP.normalLimiter_compInvMass[iP_1]);
            Vf j_normalLimiter_dstDisplacingVelocity_1 = _mm256_load_ps(&jointP.normalLimiter_dstDisplacingVelocity[iP_1]);
            Vf j_normalLimiter_accumulatedDisplacingImpulse_1 = _mm256_load_ps(&jointP.normalLimiter_accumulatedDisplacingImpulse[iP_1]);

            Vf dV0_0 = j_normalLimiter_dstDisplacingVelocity_0;

            dV0_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1X_0, body1_displacingVelocityX_0, dV0_0);
            dV0_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1Y_0, body1_displacingVelocityY_0, dV0_0);
            dV0_0 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector1_0, body1_displacingAngularVelocity_0, dV0_0);

            Vf dV1_0 = zero;

            dV1_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2X_0, body2_displacingVelocityX_0, dV1_0);
            dV1_0 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2Y_0, body2_displacingVelocityY_0, dV1_0);
            dV1_0 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector2_0, body2_displacingAngularVelocity_0, dV1_0);

            Vf dV_0 = _mm256_add_ps(dV0_0, dV1_0);

            Vf displacingDeltaImpulse_0 = _mm256_mul_ps(dV_0, j_normalLimiter_compInvMass_0);

            displacingDeltaImpulse_0 = _mm256_max_ps(displacingDeltaImpulse_0, _mm256_xor_ps(sign, j_normalLimiter_accumulatedDisplacingImpulse_0));

            body1_displacingVelocityX_0 = _mm256_fmadd_ps(j_normalLimiter_compMass1_linearX_0, displacingDeltaImpulse_0, body1_displacingVelocityX_0);
            body1_displacingVelocityY_0 = _mm256_fmadd_ps(j_normalLimiter_compMass1_linearY_0, displacingDeltaImpulse_0, body1_displacingVelocityY_0);
            body1_displacingAngularVelocity_0 = _mm256_fmadd_ps(j_normalLimiter_compMass1_angular_0, displacingDeltaImpulse_0, body1_displacingAngularVelocity_0);

            body2_displacingVelocityX_0 = _mm256_fmadd_ps(j_normalLimiter_compMass2_linearX_0, displacingDeltaImpulse_0, body2_displacingVelocityX_0);
            body2_displacingVelocityY_0 = _mm256_fmadd_ps(j_normalLimiter_compMass2_linearY_0, displacingDeltaImpulse_0, body2_displacingVelocityY_0);
            body2_displacingAngularVelocity_0 = _mm256_fmadd_ps(j_normalLimiter_compMass2_angular_0, displacingDeltaImpulse_0, body2_displacingAngularVelocity_0);

            j_normalLimiter_accumulatedDisplacingImpulse_0 = _mm256_add_ps(j_normalLimiter_accumulatedDisplacingImpulse_0, displacingDeltaImpulse_0);

            Vf dV0_1 = j_normalLimiter_dstDisplacingVelocity_1;

            dV0_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1X_1, body1_displacingVelocityX_1, dV0_1);
            dV0_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector1Y_1, body1_displacingVelocityY_1, dV0_1);
            dV0_1 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector1_1, body1_displacingAngularVelocity_1, dV0_1);

            Vf dV1_1 = zero;

            dV1_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2X_1, body2_displacingVelocityX_1, dV1_1);
            dV1_1 = _mm256_fnmadd_ps(j_normalLimiter_normalProjector2Y_1, body2_displacingVelocityY_1, dV1_1);
            dV1_1 = _mm256_fnmadd_ps(j_normalLimiter_angularProjector2_1, body2_displacingAngularVelocity_1, dV1_1);

            Vf dV_1 = _mm256_add_ps(dV0_1, dV1_1);

            Vf displacingDeltaImpulse_1 = _mm256_mul_ps(dV_1, j_normalLimiter_compInvMass_1);

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

            Vf productive_0 = _mm256_cmp_ps(_mm256_andnot_ps(sign, displacingDeltaImpulse_0), _mm256_set1_ps(kProductiveImpulse), _CMP_GT_OQ);
            Vf productive_1 = _mm256_cmp_ps(_mm256_andnot_ps(sign, displacingDeltaImpulse_1), _mm256_set1_ps(kProductiveImpulse), _CMP_GT_OQ);

            body1_lastDisplacementIteration_0 = _mm256_blendv_epi8(body1_lastDisplacementIteration_0, iterationIndex0, productive_0);
            body2_lastDisplacementIteration_0 = _mm256_blendv_epi8(body2_lastDisplacementIteration_0, iterationIndex0, productive_0);

            body1_lastDisplacementIteration_1 = _mm256_blendv_epi8(body1_lastDisplacementIteration_1, iterationIndex0, productive_1);
            body2_lastDisplacementIteration_1 = _mm256_blendv_epi8(body2_lastDisplacementIteration_1, iterationIndex0, productive_1);

            // this is a bit painful :(
            static_assert(offsetof(SolveBody, displacingVelocity) == 16 && offsetof(SolveBody, displacingAngularVelocity) == 24, "Storing assumes fixed layout");

            row0 = body1_displacingVelocityX_0;
            row1 = body1_displacingVelocityY_0;
            row2 = body1_displacingAngularVelocity_0;
            row3 = body1_lastDisplacementIteration_0;

            row4 = body2_displacingVelocityX_0;
            row5 = body2_displacingVelocityY_0;
            row6 = body2_displacingAngularVelocity_0;
            row7 = body2_lastDisplacementIteration_0;

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 0]].displacingVelocity.x, _mm256_extractf128_ps(row0, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 0]].displacingVelocity.x, _mm256_extractf128_ps(row0, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 1]].displacingVelocity.x, _mm256_extractf128_ps(row1, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 1]].displacingVelocity.x, _mm256_extractf128_ps(row1, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 2]].displacingVelocity.x, _mm256_extractf128_ps(row2, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 2]].displacingVelocity.x, _mm256_extractf128_ps(row2, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 3]].displacingVelocity.x, _mm256_extractf128_ps(row3, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 3]].displacingVelocity.x, _mm256_extractf128_ps(row3, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 4]].displacingVelocity.x, _mm256_extractf128_ps(row4, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 4]].displacingVelocity.x, _mm256_extractf128_ps(row4, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 5]].displacingVelocity.x, _mm256_extractf128_ps(row5, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 5]].displacingVelocity.x, _mm256_extractf128_ps(row5, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 6]].displacingVelocity.x, _mm256_extractf128_ps(row6, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 6]].displacingVelocity.x, _mm256_extractf128_ps(row6, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_0 + 7]].displacingVelocity.x, _mm256_extractf128_ps(row7, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_0 + 7]].displacingVelocity.x, _mm256_extractf128_ps(row7, 1));

            row0 = body1_displacingVelocityX_1;
            row1 = body1_displacingVelocityY_1;
            row2 = body1_displacingAngularVelocity_1;
            row3 = body1_lastDisplacementIteration_1;

            row4 = body2_displacingVelocityX_1;
            row5 = body2_displacingVelocityY_1;
            row6 = body2_displacingAngularVelocity_1;
            row7 = body2_lastDisplacementIteration_1;

            _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 0]].displacingVelocity.x, _mm256_extractf128_ps(row0, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 0]].displacingVelocity.x, _mm256_extractf128_ps(row0, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 1]].displacingVelocity.x, _mm256_extractf128_ps(row1, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 1]].displacingVelocity.x, _mm256_extractf128_ps(row1, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 2]].displacingVelocity.x, _mm256_extractf128_ps(row2, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 2]].displacingVelocity.x, _mm256_extractf128_ps(row2, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 3]].displacingVelocity.x, _mm256_extractf128_ps(row3, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 3]].displacingVelocity.x, _mm256_extractf128_ps(row3, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 4]].displacingVelocity.x, _mm256_extractf128_ps(row4, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 4]].displacingVelocity.x, _mm256_extractf128_ps(row4, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 5]].displacingVelocity.x, _mm256_extractf128_ps(row5, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 5]].displacingVelocity.x, _mm256_extractf128_ps(row5, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 6]].displacingVelocity.x, _mm256_extractf128_ps(row6, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 6]].displacingVelocity.x, _mm256_extractf128_ps(row6, 1));

            _mm_store_ps(&solveBodies[jointP.body1Index[iP_1 + 7]].displacingVelocity.x, _mm256_extractf128_ps(row7, 0));
            _mm_store_ps(&solveBodies[jointP.body2Index[iP_1 + 7]].displacingVelocity.x, _mm256_extractf128_ps(row7, 1));
        }
    }
#endif

    struct SolveBody
    {
        Vector2f velocity;
        float angularVelocity;

        int lastIteration;

        Vector2f displacingVelocity;
        float displacingAngularVelocity;

        int lastDisplacementIteration;
    };

    AlignedArray<SolveBody> solveBodies;

    std::vector<ContactJoint> contactJoints;

    AlignedArray<int> jointGroup_bodies;
    AlignedArray<int> jointGroup_joints;

    AlignedArray<int> joint_index;

    AlignedArray<ContactJointPacked<4>> joint_packed4;
    AlignedArray<ContactJointPacked<8>> joint_packed8;
    AlignedArray<ContactJointPacked<16>> joint_packed16;

    AlignedArray<int> joint_body1Index;
    AlignedArray<int> joint_body2Index;

    AlignedArray<float> joint_normalLimiter_normalProjector1X;
    AlignedArray<float> joint_normalLimiter_normalProjector1Y;
    AlignedArray<float> joint_normalLimiter_normalProjector2X;
    AlignedArray<float> joint_normalLimiter_normalProjector2Y;
    AlignedArray<float> joint_normalLimiter_angularProjector1;
    AlignedArray<float> joint_normalLimiter_angularProjector2;

    AlignedArray<float> joint_normalLimiter_compMass1_linearX;
    AlignedArray<float> joint_normalLimiter_compMass1_linearY;
    AlignedArray<float> joint_normalLimiter_compMass2_linearX;
    AlignedArray<float> joint_normalLimiter_compMass2_linearY;
    AlignedArray<float> joint_normalLimiter_compMass1_angular;
    AlignedArray<float> joint_normalLimiter_compMass2_angular;
    AlignedArray<float> joint_normalLimiter_compInvMass;
    AlignedArray<float> joint_normalLimiter_accumulatedImpulse;

    AlignedArray<float> joint_normalLimiter_dstVelocity;
    AlignedArray<float> joint_normalLimiter_dstDisplacingVelocity;
    AlignedArray<float> joint_normalLimiter_accumulatedDisplacingImpulse;

    AlignedArray<float> joint_frictionLimiter_normalProjector1X;
    AlignedArray<float> joint_frictionLimiter_normalProjector1Y;
    AlignedArray<float> joint_frictionLimiter_normalProjector2X;
    AlignedArray<float> joint_frictionLimiter_normalProjector2Y;
    AlignedArray<float> joint_frictionLimiter_angularProjector1;
    AlignedArray<float> joint_frictionLimiter_angularProjector2;

    AlignedArray<float> joint_frictionLimiter_compMass1_linearX;
    AlignedArray<float> joint_frictionLimiter_compMass1_linearY;
    AlignedArray<float> joint_frictionLimiter_compMass2_linearX;
    AlignedArray<float> joint_frictionLimiter_compMass2_linearY;
    AlignedArray<float> joint_frictionLimiter_compMass1_angular;
    AlignedArray<float> joint_frictionLimiter_compMass2_angular;
    AlignedArray<float> joint_frictionLimiter_compInvMass;
    AlignedArray<float> joint_frictionLimiter_accumulatedImpulse;
};