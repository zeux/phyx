#include "Joints.h"
#include <assert.h>
#include <vector>

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

class WorkQueue;

struct Solver
{
    Solver()
    {
    }

    void RefreshJoints(WorkQueue& queue);
    void PreStepJoints();

    float SolveJoints(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount);
    float SolveJointsAoS(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount);

    float SolveJointsSoA_Scalar(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount);
    float SolveJointsSoA_SSE2(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount);
    float SolveJointsSoA_AVX2(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount);

    float SolveJointsSoAPacked_Scalar(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount);
    float SolveJointsSoAPacked_SSE2(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount);
    float SolveJointsSoAPacked_AVX2(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount);
    float SolveJointsSoAPacked_FMA(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount);

    int SolvePrepareIndicesSoA(int bodiesCount, int groupSizeTarget);

    void SolvePrepareAoS(RigidBody* bodies, int bodiesCount);
    float SolveFinishAoS();

    int SolvePrepareSoA(RigidBody* bodies, int bodiesCount, int groupSizeTarget);
    float SolveFinishSoA(RigidBody* bodies, int bodiesCount);

    template <int N>
    int SolvePrepareSoAPacked(AlignedArray<ContactJointPacked<N>>& joint_packed, RigidBody* bodies, int bodiesCount, int groupSizeTarget);
    template <int N>
    float SolveFinishSoAPacked(AlignedArray<ContactJointPacked<N>>& joint_packed, RigidBody* bodies, int bodiesCount);

    bool SolveJointsImpulsesAoS(int jointStart, int jointCount, int iterationIndex);
    bool SolveJointsImpulsesSoA(int jointStart, int jointCount, int iterationIndex);
    bool SolveJointsImpulsesSoA_SSE2(int jointStart, int jointCount, int iterationIndex);
    bool SolveJointsImpulsesSoA_AVX2(int jointStart, int jointCount, int iterationIndex);

    bool SolveJointsDisplacementAoS(int jointStart, int jointCount, int iterationIndex);
    bool SolveJointsDisplacementSoA(int jointStart, int jointCount, int iterationIndex);
    bool SolveJointsDisplacementSoA_SSE2(int jointStart, int jointCount, int iterationIndex);
    bool SolveJointsDisplacementSoA_AVX2(int jointStart, int jointCount, int iterationIndex);

    template <int N>
    bool SolveJointsImpulsesSoAPacked(ContactJointPacked<N>* joint_packed, int jointStart, int jointCount, int iterationIndex);

    bool SolveJointsImpulsesSoAPacked_SSE2(ContactJointPacked<4>* joint_packed, int jointStart, int jointCount, int iterationIndex);
    bool SolveJointsImpulsesSoAPacked_AVX2(ContactJointPacked<8>* joint_packed, int jointStart, int jointCount, int iterationIndex);
    bool SolveJointsImpulsesSoAPacked_FMA(ContactJointPacked<16>* joint_packed, int jointStart, int jointCount, int iterationIndex);

    template <int N>
    bool SolveJointsDisplacementSoAPacked(ContactJointPacked<N>* joint_packed, int jointStart, int jointCount, int iterationIndex);

    bool SolveJointsDisplacementSoAPacked_SSE2(ContactJointPacked<4>* joint_packed, int jointStart, int jointCount, int iterationIndex);
    bool SolveJointsDisplacementSoAPacked_AVX2(ContactJointPacked<8>* joint_packed, int jointStart, int jointCount, int iterationIndex);
    bool SolveJointsDisplacementSoAPacked_FMA(ContactJointPacked<16>* joint_packed, int jointStart, int jointCount, int iterationIndex);

    struct SolveBody
    {
        Vector2f velocity;
        float angularVelocity;

        int lastIteration;
    };

    AlignedArray<SolveBody> solveBodiesImpulse;
    AlignedArray<SolveBody> solveBodiesDisplacement;

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