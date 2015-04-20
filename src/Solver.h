#include "Joints.h"
#include <assert.h>
#include <vector>

#include "base/AlignedArray.h"

template <int N>
struct ContactJointPacked
{
    int body1Index[N];
    int body2Index[N];

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
    Solver();

    void RefreshJoints(WorkQueue& queue);
    void PreStepJoints();

    float SolveJointsAoS(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount);

    float SolveJointsSoA_Scalar(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount);
    float SolveJointsSoA_SSE2(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount);
    float SolveJointsSoA_AVX2(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount);

    int SolvePrepareIndicesSoA(int bodiesCount, int groupSizeTarget);

    void SolvePrepareAoS(RigidBody* bodies, int bodiesCount);
    float SolveFinishAoS();

    template <int N>
    float SolveJointsSoA(AlignedArray<ContactJointPacked<N>>& joint_packed, RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount);

    template <int N>
    int SolvePrepareSoA(AlignedArray<ContactJointPacked<N>>& joint_packed, RigidBody* bodies, int bodiesCount, int groupSizeTarget);
    template <int N>
    float SolveFinishSoA(AlignedArray<ContactJointPacked<N>>& joint_packed, RigidBody* bodies, int bodiesCount);

    void PreStepJointsAoS(int jointBegin, int jointEnd);
    bool SolveJointsImpulsesAoS(int jointBegin, int jointEnd, int iterationIndex);
    bool SolveJointsDisplacementAoS(int jointBegin, int jointEnd, int iterationIndex);

    template <int VN, int N>
    void PreStepJointsSoA(ContactJointPacked<N>* joint_packed, int jointBegin, int jointEnd);
    template <int VN, int N>
    bool SolveJointsImpulsesSoA(ContactJointPacked<N>* joint_packed, int jointBegin, int jointEnd, int iterationIndex);
    template <int VN, int N>
    bool SolveJointsDisplacementSoA(ContactJointPacked<N>* joint_packed, int jointBegin, int jointEnd, int iterationIndex);

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

    AlignedArray<ContactJointPacked<1>> joint_packed1;
    AlignedArray<ContactJointPacked<4>> joint_packed4;
    AlignedArray<ContactJointPacked<8>> joint_packed8;
};