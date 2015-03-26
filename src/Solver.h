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

  NOINLINE void SolveJointsSoA(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
  {
    int groupOffset = SolvePrepareSoA(bodies, bodiesCount);

    for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
    {
      SolveJointsImpulsesSoA_AVX2(0, groupOffset);
      SolveJointsImpulsesSoA(groupOffset, contactJoints.size() - groupOffset);
    }

    for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
    {
      SolveJointsDisplacementSoA(0, groupOffset);
      SolveJointsDisplacementSoA(groupOffset, contactJoints.size() - groupOffset);
    }

    SolveFinishSoA(bodies, bodiesCount);
  }
  
  NOINLINE int SolvePrepareIndicesSoA(int bodiesCount)
  {
    int jointCount = contactJoints.size();

    jointGroup_bodies.resize(bodiesCount);
    jointGroup_joints.resize(jointCount);

    for (int i = 0; i < bodiesCount; ++i)
      jointGroup_bodies[i] = 0;

    for (int i = 0; i < jointCount; ++i)
      jointGroup_joints[i] = i;

    int tag = 0;

    int groupOffset = 0;
    int groupSizeTarget = 8;

    while (jointGroup_joints.size >= groupSizeTarget)
    {
      // gather a group of 8 joints with non-overlapping bodies
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

  NOINLINE int SolvePrepareSoA(RigidBody* bodies, int bodiesCount)
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

    int groupOffset = SolvePrepareIndicesSoA(bodiesCount);

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

  NOINLINE void SolveJointsImpulsesSoA(int jointStart, int jointCount)
  {
    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex++)
    {
      int i = jointIndex;

      SolveBody* body1 = &solveBodies[joint_body1Index[i]];
      SolveBody* body2 = &solveBodies[joint_body2Index[i]];

      {
        float dV = 0;
        dV -= joint_normalLimiter_normalProjector1X[i] * body1->velocity.x;
        dV -= joint_normalLimiter_normalProjector1Y[i] * body1->velocity.y;
        dV -= joint_normalLimiter_angularProjector1[i] * body1->angularVelocity;
        dV -= joint_normalLimiter_normalProjector2X[i] * body2->velocity.x;
        dV -= joint_normalLimiter_normalProjector2Y[i] * body2->velocity.y;
        dV -= joint_normalLimiter_angularProjector2[i] * body2->angularVelocity;
        dV += joint_normalLimiter_dstVelocity[i];

        float deltaImpulse = dV * joint_normalLimiter_compInvMass[i];

        if (deltaImpulse + joint_normalLimiter_accumulatedImpulse[i] < 0.0f)
          deltaImpulse = -joint_normalLimiter_accumulatedImpulse[i];

        body1->velocity.x += joint_normalLimiter_compMass1_linearX[i] * deltaImpulse;
        body1->velocity.y += joint_normalLimiter_compMass1_linearY[i] * deltaImpulse;
        body1->angularVelocity += joint_normalLimiter_compMass1_angular[i] * deltaImpulse;
        body2->velocity.x += joint_normalLimiter_compMass2_linearX[i] * deltaImpulse;
        body2->velocity.y += joint_normalLimiter_compMass2_linearY[i] * deltaImpulse;
        body2->angularVelocity += joint_normalLimiter_compMass2_angular[i] * deltaImpulse;

        joint_normalLimiter_accumulatedImpulse[i] += deltaImpulse;
      }

      float deltaImpulse;

      {
        float dV = 0;

        dV -= joint_frictionLimiter_normalProjector1X[i] * body1->velocity.x;
        dV -= joint_frictionLimiter_normalProjector1Y[i] * body1->velocity.y;
        dV -= joint_frictionLimiter_angularProjector1[i] * body1->angularVelocity;
        dV -= joint_frictionLimiter_normalProjector2X[i] * body2->velocity.x;
        dV -= joint_frictionLimiter_normalProjector2Y[i] * body2->velocity.y;
        dV -= joint_frictionLimiter_angularProjector2[i] * body2->angularVelocity;

        deltaImpulse = dV * joint_frictionLimiter_compInvMass[i];
      }

      float reactionForce = joint_normalLimiter_accumulatedImpulse[i];
      float accumulatedImpulse = joint_frictionLimiter_accumulatedImpulse[i];

      float frictionForce = accumulatedImpulse + deltaImpulse;
      float frictionCoefficient = 0.3f;

      if (fabsf(frictionForce) > (reactionForce * frictionCoefficient))
      {
        float dir = frictionForce > 0.0f ? 1.0f : -1.0f;
        frictionForce = dir * reactionForce * frictionCoefficient;
        deltaImpulse = frictionForce - accumulatedImpulse;
      }

      joint_frictionLimiter_accumulatedImpulse[i] += deltaImpulse;

      body1->velocity.x += joint_frictionLimiter_compMass1_linearX[i] * deltaImpulse;
      body1->velocity.y += joint_frictionLimiter_compMass1_linearY[i] * deltaImpulse;
      body1->angularVelocity += joint_frictionLimiter_compMass1_angular[i] * deltaImpulse;

      body2->velocity.x += joint_frictionLimiter_compMass2_linearX[i] * deltaImpulse;
      body2->velocity.y += joint_frictionLimiter_compMass2_linearX[i] * deltaImpulse;
      body2->angularVelocity += joint_frictionLimiter_compMass2_angular[i] * deltaImpulse;
    }
  }


  NOINLINE void SolveJointsImpulsesSoA_AVX2(int jointStart, int jointCount)
  {
    typedef __m256 Vf;
    typedef __m256i Vi;

    assert(jointStart % 8 == 0 && jointCount % 8 == 0);

    Vf sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex += 8)
    {
      int i = jointIndex;

      Vf zero = _mm256_setzero_ps();

      Vi j_body1Index = _mm256_load_si256((__m256i*)&joint_body1Index[i]);
      Vi j_body2Index = _mm256_load_si256((__m256i*)&joint_body2Index[i]);

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
      Vf j_normalLimiter_dstDisplacingVelocity = _mm256_load_ps(&joint_normalLimiter_dstDisplacingVelocity[i]);
      Vf j_normalLimiter_accumulatedDisplacingImpulse = _mm256_load_ps(&joint_normalLimiter_accumulatedDisplacingImpulse[i]);

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

      static_assert(sizeof(SolveBody) == 32, "Need to adjust bit shift below");

      Vi j_body1_offset = _mm256_slli_epi32(j_body1Index, 5);
      Vi j_body2_offset = _mm256_slli_epi32(j_body2Index, 5);

      Vf body1_velocityX = _mm256_i32gather_ps(&solveBodies[0].velocity.x, j_body1_offset, 1);
      Vf body1_velocityY = _mm256_i32gather_ps(&solveBodies[0].velocity.y, j_body1_offset, 1);
      Vf body1_angularVelocity = _mm256_i32gather_ps(&solveBodies[0].angularVelocity, j_body1_offset, 1);

      Vf body2_velocityX = _mm256_i32gather_ps(&solveBodies[0].velocity.x, j_body2_offset, 1);
      Vf body2_velocityY = _mm256_i32gather_ps(&solveBodies[0].velocity.y, j_body2_offset, 1);
      Vf body2_angularVelocity = _mm256_i32gather_ps(&solveBodies[0].angularVelocity, j_body2_offset, 1);

      {
        Vf dV = zero;

        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector1X, body1_velocityX));
        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector1Y, body1_velocityY));
        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_angularProjector1, body1_angularVelocity));

        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector2X, body2_velocityX));
        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_normalProjector2Y, body2_velocityY));
        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_normalLimiter_angularProjector2, body2_angularVelocity));

        dV = _mm256_add_ps(dV, j_normalLimiter_dstVelocity);

        Vf deltaImpulse = _mm256_mul_ps(dV, j_normalLimiter_compInvMass);

        deltaImpulse = _mm256_max_ps(deltaImpulse, _mm256_sub_ps(zero, j_normalLimiter_accumulatedImpulse));

        body1_velocityX = _mm256_add_ps(body1_velocityX, _mm256_mul_ps(j_normalLimiter_compMass1_linearX, deltaImpulse));
        body1_velocityY = _mm256_add_ps(body1_velocityY, _mm256_mul_ps(j_normalLimiter_compMass1_linearY, deltaImpulse));
        body1_angularVelocity = _mm256_add_ps(body1_angularVelocity, _mm256_mul_ps(j_normalLimiter_compMass1_angular, deltaImpulse));

        body2_velocityX = _mm256_add_ps(body2_velocityX, _mm256_mul_ps(j_normalLimiter_compMass2_linearX, deltaImpulse));
        body2_velocityY = _mm256_add_ps(body2_velocityY, _mm256_mul_ps(j_normalLimiter_compMass2_linearY, deltaImpulse));
        body2_angularVelocity = _mm256_add_ps(body2_angularVelocity, _mm256_mul_ps(j_normalLimiter_compMass2_angular, deltaImpulse));

        j_normalLimiter_accumulatedImpulse = _mm256_add_ps(j_normalLimiter_accumulatedImpulse, deltaImpulse);
      }

      Vf deltaImpulse;

      {
        Vf dV = zero;

        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_frictionLimiter_normalProjector1X, body1_velocityX));
        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_frictionLimiter_normalProjector1Y, body1_velocityY));
        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_frictionLimiter_angularProjector1, body1_angularVelocity));

        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_frictionLimiter_normalProjector2X, body2_velocityX));
        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_frictionLimiter_normalProjector2Y, body2_velocityY));
        dV = _mm256_sub_ps(dV, _mm256_mul_ps(j_frictionLimiter_angularProjector2, body2_angularVelocity));

        deltaImpulse = _mm256_mul_ps(dV, j_frictionLimiter_compInvMass);
      }

      Vf reactionForce = j_normalLimiter_accumulatedImpulse;
      Vf accumulatedImpulse = j_frictionLimiter_accumulatedImpulse;

      Vf frictionForce = _mm256_add_ps(accumulatedImpulse, deltaImpulse);
      Vf reactionForceScaled = _mm256_mul_ps(reactionForce, _mm256_set1_ps(0.3f));

      Vf frictionForceAbs = _mm256_andnot_ps(sign, frictionForce);
      Vf reactionForceScaledSigned = _mm256_xor_ps(_mm256_and_ps(frictionForce, sign), reactionForceScaled);
      Vf deltaImpulseAdjusted = _mm256_sub_ps(reactionForceScaledSigned, accumulatedImpulse);

      deltaImpulse = _mm256_blendv_ps(deltaImpulse, deltaImpulseAdjusted, _mm256_cmp_ps(frictionForceAbs, reactionForceScaled, _CMP_GT_OQ));

      j_frictionLimiter_accumulatedImpulse = _mm256_add_ps(j_frictionLimiter_accumulatedImpulse, deltaImpulse);

      body1_velocityX = _mm256_add_ps(body1_velocityX, _mm256_mul_ps(j_frictionLimiter_compMass1_linearX, deltaImpulse));
      body1_velocityY = _mm256_add_ps(body1_velocityY, _mm256_mul_ps(j_frictionLimiter_compMass1_linearY, deltaImpulse));
      body1_angularVelocity = _mm256_add_ps(body1_angularVelocity, _mm256_mul_ps(j_frictionLimiter_compMass1_angular, deltaImpulse));

      body2_velocityX = _mm256_add_ps(body2_velocityX, _mm256_mul_ps(j_frictionLimiter_compMass2_linearX, deltaImpulse));
      body2_velocityY = _mm256_add_ps(body2_velocityY, _mm256_mul_ps(j_frictionLimiter_compMass2_linearY, deltaImpulse));
      body2_angularVelocity = _mm256_add_ps(body2_angularVelocity, _mm256_mul_ps(j_frictionLimiter_compMass2_angular, deltaImpulse));

      _mm256_store_ps(&joint_normalLimiter_accumulatedImpulse[i], j_normalLimiter_accumulatedImpulse);
      _mm256_store_ps(&joint_frictionLimiter_accumulatedImpulse[i], j_frictionLimiter_accumulatedImpulse);

      // this is a bit painful :(
      static_assert(offsetof(SolveBody, velocity) == 0, "Store code assumes fixed layout");
      static_assert(offsetof(SolveBody, angularVelocity) == 8, "Store code assumes fixed layout");

      __m128 row0, row1, row2, row3;

      row0 = _mm256_extractf128_ps(body1_velocityX, 0);
      row1 = _mm256_extractf128_ps(body1_velocityY, 0);
      row2 = _mm256_extractf128_ps(body1_angularVelocity, 0);
      row3 = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

      _mm_store_ps(&solveBodies[joint_body1Index[i + 0]].velocity.x, row0);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 1]].velocity.x, row1);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 2]].velocity.x, row2);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 3]].velocity.x, row3);

      row0 = _mm256_extractf128_ps(body1_velocityX, 1);
      row1 = _mm256_extractf128_ps(body1_velocityY, 1);
      row2 = _mm256_extractf128_ps(body1_angularVelocity, 1);
      row3 = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

      _mm_store_ps(&solveBodies[joint_body1Index[i + 4]].velocity.x, row0);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 5]].velocity.x, row1);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 6]].velocity.x, row2);
      _mm_store_ps(&solveBodies[joint_body1Index[i + 7]].velocity.x, row3);

      row0 = _mm256_extractf128_ps(body2_velocityX, 0);
      row1 = _mm256_extractf128_ps(body2_velocityY, 0);
      row2 = _mm256_extractf128_ps(body2_angularVelocity, 0);
      row3 = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

      _mm_store_ps(&solveBodies[joint_body2Index[i + 0]].velocity.x, row0);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 1]].velocity.x, row1);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 2]].velocity.x, row2);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 3]].velocity.x, row3);

      row0 = _mm256_extractf128_ps(body2_velocityX, 1);
      row1 = _mm256_extractf128_ps(body2_velocityY, 1);
      row2 = _mm256_extractf128_ps(body2_angularVelocity, 1);
      row3 = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

      _mm_store_ps(&solveBodies[joint_body2Index[i + 4]].velocity.x, row0);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 5]].velocity.x, row1);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 6]].velocity.x, row2);
      _mm_store_ps(&solveBodies[joint_body2Index[i + 7]].velocity.x, row3);
    }
  }

  NOINLINE void SolveJointsDisplacementSoA(int jointStart, int jointCount)
  {
    for (int jointIndex = jointStart; jointIndex < jointStart + jointCount; jointIndex++)
    {
      int i = jointIndex;

      SolveBody* body1 = &solveBodies[joint_body1Index[i]];
      SolveBody* body2 = &solveBodies[joint_body2Index[i]];

      float dV = 0;
      dV -= joint_normalLimiter_normalProjector1X[i] * body1->displacingVelocity.x;
      dV -= joint_normalLimiter_normalProjector1Y[i] * body1->displacingVelocity.y;
      dV -= joint_normalLimiter_angularProjector1[i] * body1->displacingAngularVelocity;
      dV -= joint_normalLimiter_normalProjector2X[i] * body2->displacingVelocity.x;
      dV -= joint_normalLimiter_normalProjector2Y[i] * body2->displacingVelocity.y;
      dV -= joint_normalLimiter_angularProjector2[i] * body2->displacingAngularVelocity;
      dV += joint_normalLimiter_dstDisplacingVelocity[i];

      float deltaDisplacingImpulse = dV * joint_normalLimiter_compInvMass[i];

      if (deltaDisplacingImpulse + joint_normalLimiter_accumulatedDisplacingImpulse[i] < 0.0f)
        deltaDisplacingImpulse = -joint_normalLimiter_accumulatedDisplacingImpulse[i];

      body1->displacingVelocity.x += joint_normalLimiter_compMass1_linearX[i] * deltaDisplacingImpulse;
      body1->displacingVelocity.y += joint_normalLimiter_compMass1_linearY[i] * deltaDisplacingImpulse;
      body1->displacingAngularVelocity += joint_normalLimiter_compMass1_angular[i] * deltaDisplacingImpulse;
      body2->displacingVelocity.x += joint_normalLimiter_compMass2_linearX[i] * deltaDisplacingImpulse;
      body2->displacingVelocity.y += joint_normalLimiter_compMass2_linearY[i] * deltaDisplacingImpulse;
      body2->displacingAngularVelocity += joint_normalLimiter_compMass2_angular[i] * deltaDisplacingImpulse;

      joint_normalLimiter_accumulatedDisplacingImpulse[i] += deltaDisplacingImpulse;
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