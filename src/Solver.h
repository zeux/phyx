#include "Joints.h"
#include <assert.h>
#include <vector>

#include "base/AlignedArray.h"

template <int N>
struct ContactLimiterPacked
{
    float normalProjector1X[N];
    float normalProjector1Y[N];
    float normalProjector2X[N];
    float normalProjector2Y[N];
    float angularProjector1[N];
    float angularProjector2[N];

    float compMass1_linearX[N];
    float compMass1_linearY[N];
    float compMass2_linearX[N];
    float compMass2_linearY[N];
    float compMass1_angular[N];
    float compMass2_angular[N];
    float compInvMass[N];
};

template <int N>
struct ContactJointPacked
{
    int body1Index[N];
    int body2Index[N];
    int contactPointIndex[N];

    ContactLimiterPacked<N> normalLimiter;

    float normalLimiter_compInvMass[N];
    float normalLimiter_accumulatedImpulse[N];

    float normalLimiter_dstVelocity[N];
    float normalLimiter_dstDisplacingVelocity[N];
    float normalLimiter_accumulatedDisplacingImpulse[N];

    ContactLimiterPacked<N> frictionLimiter;

    float frictionLimiter_accumulatedImpulse[N];
};

class WorkQueue;

struct Solver
{
    Solver();

    float SolveJoints_Scalar(WorkQueue& queue, RigidBody* bodies, int bodiesCount, ContactPoint* contactPoints, int contactIterationsCount, int penetrationIterationsCount);
    float SolveJoints_SSE2(WorkQueue& queue, RigidBody* bodies, int bodiesCount, ContactPoint* contactPoints, int contactIterationsCount, int penetrationIterationsCount);
    float SolveJoints_AVX2(WorkQueue& queue, RigidBody* bodies, int bodiesCount, ContactPoint* contactPoints, int contactIterationsCount, int penetrationIterationsCount);

    int PrepareIndices(int bodiesCount, int groupSizeTarget);
    void GatherIslands(RigidBody* bodies, int bodiesCount);

    template <int N>
    float SolveJoints(WorkQueue& queue, AlignedArray<ContactJointPacked<N>>& joint_packed, RigidBody* bodies, int bodiesCount, ContactPoint* contactPoints, int contactIterationsCount, int penetrationIterationsCount);

    template <int N>
    int SolvePrepare(AlignedArray<ContactJointPacked<N>>& joint_packed, RigidBody* bodies, int bodiesCount, int groupSizeTarget);
    template <int N>
    float SolveFinish(AlignedArray<ContactJointPacked<N>>& joint_packed, RigidBody* bodies, int bodiesCount);

    template <int VN, int N>
    void RefreshJoints(ContactJointPacked<N>* joint_packed, int jointBegin, int jointEnd, ContactPoint* contactPoints);
    template <int VN, int N>
    void PreStepJoints(ContactJointPacked<N>* joint_packed, int jointBegin, int jointEnd);
    template <int VN, int N>
    bool SolveJointsImpulses(ContactJointPacked<N>* joint_packed, int jointBegin, int jointEnd, int iterationIndex);
    template <int VN, int N>
    bool SolveJointsDisplacement(ContactJointPacked<N>* joint_packed, int jointBegin, int jointEnd, int iterationIndex);

    struct SolveBodyParams
    {
        float invMass;
        float invInertia;

        Vector2f coords_pos;

        Vector2f coords_xVector;
        Vector2f coords_yVector;
    };

    struct SolveBody
    {
        Vector2f velocity;
        float angularVelocity;

        int lastIteration;
    };

    int islandCount;
    int islandMaxSize;

    AlignedArray<SolveBodyParams> solveBodiesParams;
    AlignedArray<SolveBody> solveBodiesImpulse;
    AlignedArray<SolveBody> solveBodiesDisplacement;

    std::vector<ContactJoint> contactJoints;

    AlignedArray<int> jointGroup_bodies;
    AlignedArray<int> jointGroup_joints;

    AlignedArray<int> joint_index;

    AlignedArray<int> island_remap;
    AlignedArray<int> island_index;
    AlignedArray<int> island_offset;
    AlignedArray<int> island_offsettemp;

    AlignedArray<ContactJointPacked<1>> joint_packed1;
    AlignedArray<ContactJointPacked<4>> joint_packed4;
    AlignedArray<ContactJointPacked<8>> joint_packed8;
};