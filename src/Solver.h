#include "Joints.h"
#include <assert.h>
#include <vector>

#include <immintrin.h>

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

template <typename T> struct AlignedArray
{
  T* data;
  int size;
  int capacity;

  AlignedArray(): data(0), size(0), capacity(0)
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
      aligned_free(data);

      data = static_cast<T*>(aligned_alloc(newsize * sizeof(T), 16));
      capacity = newsize;
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

struct Solver
{
  Solver()
  {
  }
  void RefreshJoints()
  {
    for (size_t jointIndex = 0; jointIndex < contactJoints.size();)
    {
      if (!contactJoints[jointIndex].valid)
      {
        contactJoints[jointIndex] = contactJoints[contactJoints.size() - 1];
        contactJoints.pop_back();
      }
      else
      {
        contactJoints[jointIndex++].Refresh();
      }
    }
    for (size_t jointIndex = 0; jointIndex < contactJoints.size(); jointIndex++)
    {
      contactJoints[jointIndex].PreStep();
    }
  }

  void RefreshContactJoint(ContactJoint::Descriptor desc)
  {
    ContactJoint *joint = (ContactJoint*)(desc.collision->userInfo);

    joint->valid = 1;
    joint->collision = desc.collision;

    assert(joint->body1 == desc.body1);
    assert(joint->body2 == desc.body2);

    if ((joint->body1->coords.pos - joint->body2->coords.pos) * desc.collision->normal < 0.0f)
    {
      int pp = 1;
    }

    joint->valid = 1;
    joint->collision = desc.collision;
  }

  NOINLINE void SolveJoints(int contactIterationsCount, int penetrationIterationsCount)
  {
    for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
    {
      for (size_t jointIndex = 0; jointIndex < contactJoints.size(); jointIndex++)
      {
        contactJoints[jointIndex].SolveImpulse();
      }
    }
    for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
    {
      for (size_t jointIndex = 0; jointIndex < contactJoints.size(); jointIndex++)
      {
        contactJoints[jointIndex].SolveDisplacement();
      }
    }
  }

  NOINLINE void SolveJointsAoS(int contactIterationsCount, int penetrationIterationsCount)
  {
    for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
    {
      SolveJointsImpulsesAoS(0, contactJoints.size());
    }

    for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
    {
      SolveJointsDisplacementAoS(0, contactJoints.size());
    }
  }

  NOINLINE void SolveJointsSoA_Scalar(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
  {
    int groupOffset = SolvePrepareSoA(bodies, bodiesCount, 1);

    for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
    {
      SolveJointsImpulsesSoA(0, contactJoints.size());
    }

    for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
    {
      SolveJointsDisplacementSoA(0, contactJoints.size());
    }

    SolveFinishSoA(bodies, bodiesCount);
  }
  
  NOINLINE void SolveJointsSoA_SSE2(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
  {
    int groupOffset = SolvePrepareSoA(bodies, bodiesCount, 4);

    for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
    {
      SolveJointsImpulsesSoA_SSE2(0, groupOffset);
      SolveJointsImpulsesSoA(groupOffset, contactJoints.size() - groupOffset);
    }

    for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
    {
      SolveJointsDisplacementSoA_SSE2(0, groupOffset);
      SolveJointsDisplacementSoA(groupOffset, contactJoints.size() - groupOffset);
    }

    SolveFinishSoA(bodies, bodiesCount);
  }
  
  NOINLINE void SolveJointsSoA_AVX2(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
  {
    int groupOffset = SolvePrepareSoA(bodies, bodiesCount, 8);

    for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
    {
      SolveJointsImpulsesSoA_AVX2(0, groupOffset);
      SolveJointsImpulsesSoA(groupOffset, contactJoints.size() - groupOffset);
    }

    for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
    {
      SolveJointsDisplacementSoA_AVX2(0, groupOffset);
      SolveJointsDisplacementSoA(groupOffset, contactJoints.size() - groupOffset);
    }

    SolveFinishSoA(bodies, bodiesCount);
  }
  
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

        for (int i = 0; i < jointGroup_joints.size && groupSize < groupSizeTarget; )
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

      solveBodies[i].displacingVelocity = bodies[i].displacingVelocity;
      solveBodies[i].displacingAngularVelocity = bodies[i].displacingAngularVelocity;
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

  NOINLINE void SolveFinishSoA(RigidBody* bodies, int bodiesCount)
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
  }

  NOINLINE void SolveJointsImpulsesAoS(int jointStart, int jointCount)
  {
    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex++)
    {
      ContactJoint& joint = contactJoints[jointIndex];

      RigidBody* body1 = joint.body1;
      RigidBody* body2 = joint.body2;

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
    }
  }

  NOINLINE void SolveJointsImpulsesSoA(int jointStart, int jointCount)
  {
    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex++)
    {
      int i = jointIndex;

      SolveBody* body1 = &solveBodies[joint_body1Index[i]];
      SolveBody* body2 = &solveBodies[joint_body2Index[i]];

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
      body2->velocity.y += joint_frictionLimiter_compMass2_linearX[i] * frictionDeltaImpulse;
      body2->angularVelocity += joint_frictionLimiter_compMass2_angular[i] * frictionDeltaImpulse;
    }
  }

  NOINLINE void SolveJointsImpulsesSoA_SSE2(int jointStart, int jointCount)
  {
    typedef __m128 Vf;

    assert(jointStart % 4 == 0 && jointCount % 4 == 0);

    Vf sign = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));

    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 4)
    {
      int i = jointIndex;

      Vf zero = _mm_setzero_ps();

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

      row0 = _mm_load_ps(&solveBodies[joint_body2Index[i + 0]].velocity.x);
      row1 = _mm_load_ps(&solveBodies[joint_body2Index[i + 1]].velocity.x);
      row2 = _mm_load_ps(&solveBodies[joint_body2Index[i + 2]].velocity.x);
      row3 = _mm_load_ps(&solveBodies[joint_body2Index[i + 3]].velocity.x);

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

      Vf body2_velocityX = row0;
      Vf body2_velocityY = row1;
      Vf body2_angularVelocity = row2;

      Vf normaldV = j_normalLimiter_dstVelocity;

      normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_normalProjector1X, body1_velocityX));
      normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_normalProjector1Y, body1_velocityY));
      normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_angularProjector1, body1_angularVelocity));

      normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_normalProjector2X, body2_velocityX));
      normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_normalProjector2Y, body2_velocityY));
      normaldV = _mm_sub_ps(normaldV, _mm_mul_ps(j_normalLimiter_angularProjector2, body2_angularVelocity));

      Vf normalDeltaImpulse = _mm_mul_ps(normaldV, j_normalLimiter_compInvMass);

      normalDeltaImpulse = _mm_max_ps(normalDeltaImpulse, _mm_sub_ps(zero, j_normalLimiter_accumulatedImpulse));

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

      // this is a bit painful :(
      static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Storing assumes fixed layout");

      row0 = body1_velocityX;
      row1 = body1_velocityY;
      row2 = body1_angularVelocity;
      row3 = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

      _mm_store_ps(&solveBodies[joint_body1Index[i + 0]].velocity.x, row0);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 1]].velocity.x, row1);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 2]].velocity.x, row2);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 3]].velocity.x, row3);

      row0 = body2_velocityX;
      row1 = body2_velocityY;
      row2 = body2_angularVelocity;
      row3 = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

      _mm_store_ps(&solveBodies[joint_body2Index[i + 0]].velocity.x, row0);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 1]].velocity.x, row1);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 2]].velocity.x, row2);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 3]].velocity.x, row3);
    }
  }

  static inline __m256 _mm256_combine_ps(__m128 a, __m128 b)
  {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
  }

  NOINLINE void SolveJointsImpulsesSoA_AVX2(int jointStart, int jointCount)
  {
    typedef __m256 Vf;

    assert(jointStart % 8 == 0 && jointCount % 8 == 0);

    Vf sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 8)
    {
      int i = jointIndex;

      Vf zero = _mm256_setzero_ps();

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

      __m128 row0, row1, row2, row3, row4, row5, row6, row7;

      static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Loading assumes fixed layout");

      row0 = _mm_load_ps(&solveBodies[joint_body1Index[i + 0]].velocity.x);
      row1 = _mm_load_ps(&solveBodies[joint_body1Index[i + 1]].velocity.x);
      row2 = _mm_load_ps(&solveBodies[joint_body1Index[i + 2]].velocity.x);
      row3 = _mm_load_ps(&solveBodies[joint_body1Index[i + 3]].velocity.x);

      row4 = _mm_load_ps(&solveBodies[joint_body1Index[i + 4]].velocity.x);
      row5 = _mm_load_ps(&solveBodies[joint_body1Index[i + 5]].velocity.x);
      row6 = _mm_load_ps(&solveBodies[joint_body1Index[i + 6]].velocity.x);
      row7 = _mm_load_ps(&solveBodies[joint_body1Index[i + 7]].velocity.x);

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
      _MM_TRANSPOSE4_PS(row4, row5, row6, row7);

      Vf body1_velocityX = _mm256_combine_ps(row0, row4);
      Vf body1_velocityY = _mm256_combine_ps(row1, row5);
      Vf body1_angularVelocity = _mm256_combine_ps(row2, row6);

      row0 = _mm_load_ps(&solveBodies[joint_body2Index[i + 0]].velocity.x);
      row1 = _mm_load_ps(&solveBodies[joint_body2Index[i + 1]].velocity.x);
      row2 = _mm_load_ps(&solveBodies[joint_body2Index[i + 2]].velocity.x);
      row3 = _mm_load_ps(&solveBodies[joint_body2Index[i + 3]].velocity.x);

      row4 = _mm_load_ps(&solveBodies[joint_body2Index[i + 4]].velocity.x);
      row5 = _mm_load_ps(&solveBodies[joint_body2Index[i + 5]].velocity.x);
      row6 = _mm_load_ps(&solveBodies[joint_body2Index[i + 6]].velocity.x);
      row7 = _mm_load_ps(&solveBodies[joint_body2Index[i + 7]].velocity.x);

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
      _MM_TRANSPOSE4_PS(row4, row5, row6, row7);

      Vf body2_velocityX = _mm256_combine_ps(row0, row4);
      Vf body2_velocityY = _mm256_combine_ps(row1, row5);
      Vf body2_angularVelocity = _mm256_combine_ps(row2, row6);

      Vf normaldV = j_normalLimiter_dstVelocity;

      normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector1X, body1_velocityX));
      normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector1Y, body1_velocityY));
      normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_angularProjector1, body1_angularVelocity));

      normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector2X, body2_velocityX));
      normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_normalProjector2Y, body2_velocityY));
      normaldV = _mm256_sub_ps(normaldV, _mm256_mul_ps(j_normalLimiter_angularProjector2, body2_angularVelocity));

      Vf normalDeltaImpulse = _mm256_mul_ps(normaldV, j_normalLimiter_compInvMass);

      normalDeltaImpulse = _mm256_max_ps(normalDeltaImpulse, _mm256_sub_ps(zero, j_normalLimiter_accumulatedImpulse));

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

      // this is a bit painful :(
      static_assert(offsetof(SolveBody, velocity) == 0 && offsetof(SolveBody, angularVelocity) == 8, "Storing assumes fixed layout");

      row0 = _mm256_extractf128_ps(body1_velocityX, 0);
      row1 = _mm256_extractf128_ps(body1_velocityY, 0);
      row2 = _mm256_extractf128_ps(body1_angularVelocity, 0);
      row3 = _mm_setzero_ps();

      row4 = _mm256_extractf128_ps(body1_velocityX, 1);
      row5 = _mm256_extractf128_ps(body1_velocityY, 1);
      row6 = _mm256_extractf128_ps(body1_angularVelocity, 1);
      row7 = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
      _MM_TRANSPOSE4_PS(row4, row5, row6, row7);

      _mm_store_ps(&solveBodies[joint_body1Index[i + 0]].velocity.x, row0);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 1]].velocity.x, row1);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 2]].velocity.x, row2);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 3]].velocity.x, row3);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 4]].velocity.x, row4);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 5]].velocity.x, row5);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 6]].velocity.x, row6);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 7]].velocity.x, row7);

      row0 = _mm256_extractf128_ps(body2_velocityX, 0);
      row1 = _mm256_extractf128_ps(body2_velocityY, 0);
      row2 = _mm256_extractf128_ps(body2_angularVelocity, 0);
      row3 = _mm_setzero_ps();

      row4 = _mm256_extractf128_ps(body2_velocityX, 1);
      row5 = _mm256_extractf128_ps(body2_velocityY, 1);
      row6 = _mm256_extractf128_ps(body2_angularVelocity, 1);
      row7 = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
      _MM_TRANSPOSE4_PS(row4, row5, row6, row7);

      _mm_store_ps(&solveBodies[joint_body2Index[i + 0]].velocity.x, row0);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 1]].velocity.x, row1);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 2]].velocity.x, row2);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 3]].velocity.x, row3);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 4]].velocity.x, row4);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 5]].velocity.x, row5);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 6]].velocity.x, row6);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 7]].velocity.x, row7);
    }
  }

  NOINLINE void SolveJointsDisplacementAoS(int jointStart, int jointCount)
  {
    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex++)
    {
      ContactJoint& joint = contactJoints[jointIndex];

      RigidBody* body1 = joint.body1;
      RigidBody* body2 = joint.body2;

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
    }
  }

  NOINLINE void SolveJointsDisplacementSoA(int jointStart, int jointCount)
  {
    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex++)
    {
      int i = jointIndex;

      SolveBody* body1 = &solveBodies[joint_body1Index[i]];
      SolveBody* body2 = &solveBodies[joint_body2Index[i]];

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
    }
  }

  NOINLINE void SolveJointsDisplacementSoA_SSE2(int jointStart, int jointCount)
  {
    typedef __m128 Vf;

    assert(jointStart % 4 == 0 && jointCount % 4 == 0);

    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 4)
    {
      int i = jointIndex;

      Vf zero = _mm_setzero_ps();

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

      row0 = _mm_load_ps(&solveBodies[joint_body2Index[i + 0]].displacingVelocity.x);
      row1 = _mm_load_ps(&solveBodies[joint_body2Index[i + 1]].displacingVelocity.x);
      row2 = _mm_load_ps(&solveBodies[joint_body2Index[i + 2]].displacingVelocity.x);
      row3 = _mm_load_ps(&solveBodies[joint_body2Index[i + 3]].displacingVelocity.x);

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

      Vf body2_displacingVelocityX = row0;
      Vf body2_displacingVelocityY = row1;
      Vf body2_displacingAngularVelocity = row2;

      Vf dV = j_normalLimiter_dstDisplacingVelocity;

      dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_normalProjector1X, body1_displacingVelocityX));
      dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_normalProjector1Y, body1_displacingVelocityY));
      dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_angularProjector1, body1_displacingAngularVelocity));

      dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_normalProjector2X, body2_displacingVelocityX));
      dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_normalProjector2Y, body2_displacingVelocityY));
      dV = _mm_sub_ps(dV, _mm_mul_ps(j_normalLimiter_angularProjector2, body2_displacingAngularVelocity));

      Vf displacingDeltaImpulse = _mm_mul_ps(dV, j_normalLimiter_compInvMass);

      displacingDeltaImpulse = _mm_max_ps(displacingDeltaImpulse, _mm_sub_ps(zero, j_normalLimiter_accumulatedDisplacingImpulse));

      body1_displacingVelocityX = _mm_add_ps(body1_displacingVelocityX, _mm_mul_ps(j_normalLimiter_compMass1_linearX, displacingDeltaImpulse));
      body1_displacingVelocityY = _mm_add_ps(body1_displacingVelocityY, _mm_mul_ps(j_normalLimiter_compMass1_linearY, displacingDeltaImpulse));
      body1_displacingAngularVelocity = _mm_add_ps(body1_displacingAngularVelocity, _mm_mul_ps(j_normalLimiter_compMass1_angular, displacingDeltaImpulse));

      body2_displacingVelocityX = _mm_add_ps(body2_displacingVelocityX, _mm_mul_ps(j_normalLimiter_compMass2_linearX, displacingDeltaImpulse));
      body2_displacingVelocityY = _mm_add_ps(body2_displacingVelocityY, _mm_mul_ps(j_normalLimiter_compMass2_linearY, displacingDeltaImpulse));
      body2_displacingAngularVelocity = _mm_add_ps(body2_displacingAngularVelocity, _mm_mul_ps(j_normalLimiter_compMass2_angular, displacingDeltaImpulse));

      j_normalLimiter_accumulatedDisplacingImpulse = _mm_add_ps(j_normalLimiter_accumulatedDisplacingImpulse, displacingDeltaImpulse);

      _mm_store_ps(&joint_normalLimiter_accumulatedDisplacingImpulse[i], j_normalLimiter_accumulatedDisplacingImpulse);

      // this is a bit painful :(
      static_assert(offsetof(SolveBody, displacingVelocity) == 16 && offsetof(SolveBody, displacingAngularVelocity) == 24, "Storing assumes fixed layout");

      row0 = body1_displacingVelocityX;
      row1 = body1_displacingVelocityY;
      row2 = body1_displacingAngularVelocity;
      row3 = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

      _mm_store_ps(&solveBodies[joint_body1Index[i + 0]].displacingVelocity.x, row0);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 1]].displacingVelocity.x, row1);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 2]].displacingVelocity.x, row2);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 3]].displacingVelocity.x, row3);

      row0 = body2_displacingVelocityX;
      row1 = body2_displacingVelocityY;
      row2 = body2_displacingAngularVelocity;
      row3 = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

      _mm_store_ps(&solveBodies[joint_body2Index[i + 0]].displacingVelocity.x, row0);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 1]].displacingVelocity.x, row1);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 2]].displacingVelocity.x, row2);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 3]].displacingVelocity.x, row3);
    }
  }

  NOINLINE void SolveJointsDisplacementSoA_AVX2(int jointStart, int jointCount)
  {
    typedef __m256 Vf;

    assert(jointStart % 8 == 0 && jointCount % 8 == 0);

    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 8)
    {
      int i = jointIndex;

      Vf zero = _mm256_setzero_ps();

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

      __m128 row0, row1, row2, row3, row4, row5, row6, row7;

      static_assert(offsetof(SolveBody, displacingVelocity) == 16 && offsetof(SolveBody, displacingAngularVelocity) == 24, "Loading assumes fixed layout");

      row0 = _mm_load_ps(&solveBodies[joint_body1Index[i + 0]].displacingVelocity.x);
      row1 = _mm_load_ps(&solveBodies[joint_body1Index[i + 1]].displacingVelocity.x);
      row2 = _mm_load_ps(&solveBodies[joint_body1Index[i + 2]].displacingVelocity.x);
      row3 = _mm_load_ps(&solveBodies[joint_body1Index[i + 3]].displacingVelocity.x);

      row4 = _mm_load_ps(&solveBodies[joint_body1Index[i + 4]].displacingVelocity.x);
      row5 = _mm_load_ps(&solveBodies[joint_body1Index[i + 5]].displacingVelocity.x);
      row6 = _mm_load_ps(&solveBodies[joint_body1Index[i + 6]].displacingVelocity.x);
      row7 = _mm_load_ps(&solveBodies[joint_body1Index[i + 7]].displacingVelocity.x);

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
      _MM_TRANSPOSE4_PS(row4, row5, row6, row7);

      Vf body1_displacingVelocityX = _mm256_combine_ps(row0, row4);
      Vf body1_displacingVelocityY = _mm256_combine_ps(row1, row5);
      Vf body1_displacingAngularVelocity = _mm256_combine_ps(row2, row6);

      row0 = _mm_load_ps(&solveBodies[joint_body2Index[i + 0]].displacingVelocity.x);
      row1 = _mm_load_ps(&solveBodies[joint_body2Index[i + 1]].displacingVelocity.x);
      row2 = _mm_load_ps(&solveBodies[joint_body2Index[i + 2]].displacingVelocity.x);
      row3 = _mm_load_ps(&solveBodies[joint_body2Index[i + 3]].displacingVelocity.x);

      row4 = _mm_load_ps(&solveBodies[joint_body2Index[i + 4]].displacingVelocity.x);
      row5 = _mm_load_ps(&solveBodies[joint_body2Index[i + 5]].displacingVelocity.x);
      row6 = _mm_load_ps(&solveBodies[joint_body2Index[i + 6]].displacingVelocity.x);
      row7 = _mm_load_ps(&solveBodies[joint_body2Index[i + 7]].displacingVelocity.x);

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
      _MM_TRANSPOSE4_PS(row4, row5, row6, row7);

      Vf body2_displacingVelocityX = _mm256_combine_ps(row0, row4);
      Vf body2_displacingVelocityY = _mm256_combine_ps(row1, row5);
      Vf body2_displacingAngularVelocity = _mm256_combine_ps(row2, row6);

      Vf dV = j_normalLimiter_dstDisplacingVelocity;

      dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector1X, body1_displacingVelocityX));
      dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector1Y, body1_displacingVelocityY));
      dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_angularProjector1, body1_displacingAngularVelocity));

      dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector2X, body2_displacingVelocityX));
      dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector2Y, body2_displacingVelocityY));
      dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_angularProjector2, body2_displacingAngularVelocity));

      Vf displacingDeltaImpulse = _mm256_mul_ps(dV, j_normalLimiter_compInvMass);

      displacingDeltaImpulse = _mm256_max_ps(displacingDeltaImpulse, _mm256_sub_ps(zero, j_normalLimiter_accumulatedDisplacingImpulse));

      body1_displacingVelocityX = _mm256_add_ps(body1_displacingVelocityX, _mm256_mul_ps(j_normalLimiter_compMass1_linearX, displacingDeltaImpulse));
      body1_displacingVelocityY = _mm256_add_ps(body1_displacingVelocityY, _mm256_mul_ps(j_normalLimiter_compMass1_linearY, displacingDeltaImpulse));
      body1_displacingAngularVelocity = _mm256_add_ps(body1_displacingAngularVelocity, _mm256_mul_ps(j_normalLimiter_compMass1_angular, displacingDeltaImpulse));

      body2_displacingVelocityX = _mm256_add_ps(body2_displacingVelocityX, _mm256_mul_ps(j_normalLimiter_compMass2_linearX, displacingDeltaImpulse));
      body2_displacingVelocityY = _mm256_add_ps(body2_displacingVelocityY, _mm256_mul_ps(j_normalLimiter_compMass2_linearY, displacingDeltaImpulse));
      body2_displacingAngularVelocity = _mm256_add_ps(body2_displacingAngularVelocity, _mm256_mul_ps(j_normalLimiter_compMass2_angular, displacingDeltaImpulse));

      j_normalLimiter_accumulatedDisplacingImpulse = _mm256_add_ps(j_normalLimiter_accumulatedDisplacingImpulse, displacingDeltaImpulse);

      _mm256_store_ps(&joint_normalLimiter_accumulatedDisplacingImpulse[i], j_normalLimiter_accumulatedDisplacingImpulse);

      // this is a bit painful :(
      static_assert(offsetof(SolveBody, displacingVelocity) == 16 && offsetof(SolveBody, displacingAngularVelocity) == 24, "Storing assumes fixed layout");

      row0 = _mm256_extractf128_ps(body1_displacingVelocityX, 0);
      row1 = _mm256_extractf128_ps(body1_displacingVelocityY, 0);
      row2 = _mm256_extractf128_ps(body1_displacingAngularVelocity, 0);
      row3 = _mm_setzero_ps();

      row4 = _mm256_extractf128_ps(body1_displacingVelocityX, 1);
      row5 = _mm256_extractf128_ps(body1_displacingVelocityY, 1);
      row6 = _mm256_extractf128_ps(body1_displacingAngularVelocity, 1);
      row7 = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
      _MM_TRANSPOSE4_PS(row4, row5, row6, row7);

      _mm_store_ps(&solveBodies[joint_body1Index[i + 0]].displacingVelocity.x, row0);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 1]].displacingVelocity.x, row1);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 2]].displacingVelocity.x, row2);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 3]].displacingVelocity.x, row3);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 4]].displacingVelocity.x, row4);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 5]].displacingVelocity.x, row5);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 6]].displacingVelocity.x, row6);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 7]].displacingVelocity.x, row7);

      row0 = _mm256_extractf128_ps(body2_displacingVelocityX, 0);
      row1 = _mm256_extractf128_ps(body2_displacingVelocityY, 0);
      row2 = _mm256_extractf128_ps(body2_displacingAngularVelocity, 0);
      row3 = _mm_setzero_ps();

      row4 = _mm256_extractf128_ps(body2_displacingVelocityX, 1);
      row5 = _mm256_extractf128_ps(body2_displacingVelocityY, 1);
      row6 = _mm256_extractf128_ps(body2_displacingAngularVelocity, 1);
      row7 = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
      _MM_TRANSPOSE4_PS(row4, row5, row6, row7);

      _mm_store_ps(&solveBodies[joint_body2Index[i + 0]].displacingVelocity.x, row0);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 1]].displacingVelocity.x, row1);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 2]].displacingVelocity.x, row2);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 3]].displacingVelocity.x, row3);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 4]].displacingVelocity.x, row4);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 5]].displacingVelocity.x, row5);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 6]].displacingVelocity.x, row6);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 7]].displacingVelocity.x, row7);
    }
  }

  struct SolveBody
  {
   Vector2f velocity;
   float angularVelocity;

   float padding1;

   Vector2f displacingVelocity;
   float displacingAngularVelocity;

   float padding2;
 };

 AlignedArray<SolveBody> solveBodies;

 std::vector<ContactJoint> contactJoints;

 AlignedArray<int> jointGroup_bodies;
 AlignedArray<int> jointGroup_joints;

 AlignedArray<int> joint_index;
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