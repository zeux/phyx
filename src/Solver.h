#include "Joints.h"
#include <assert.h>
#include <vector>

#include <emmintrin.h>

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

  NOINLINE void SolveJoints(RigidBody* bodies, int bodiesCount, int contactIterationsCount, int penetrationIterationsCount)
  {
    SolvePrepare(bodies, bodiesCount);

    for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
    {
      SolveJointsImpulses();
    }

    for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
    {
      SolveJointsDisplacement();
    }

    SolveFinish(bodies, bodiesCount);
  }
  
  NOINLINE void SolvePrepare(RigidBody* bodies, int bodiesCount)
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

    for (int i = 0; i < jointCount; ++i)
    {
      ContactJoint& joint = contactJoints[i];

      joint_index[i] = i;
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
  }

  NOINLINE void SolveFinish(RigidBody* bodies, int bodiesCount)
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
      ContactJoint& joint = contactJoints[i];

    /*
      joint.normalLimiter.accumulatedImpulse = joint_normalLimiter_accumulatedImpulse[i];
      joint.normalLimiter.accumulatedDisplacingImpulse = joint_normalLimiter_accumulatedDisplacingImpulse[i];
      joint.frictionLimiter.accumulatedImpulse = joint_frictionLimiter_accumulatedImpulse[i];
    */
    }
  }

  NOINLINE void SolveJointsImpulses()
  {
      for (size_t jointIndex = 0; jointIndex < contactJoints.size(); jointIndex++)
      {
        ContactJoint& joint = contactJoints[jointIndex];

        SolveBody* body1 = &solveBodies[joint.body1Index];
        SolveBody* body2 = &solveBodies[joint.body2Index];

        {
          float dV = 0;
          dV -= joint.normalLimiter.normalProjector1 * body1->velocity;
          dV -= joint.normalLimiter.angularProjector1 * body1->angularVelocity;
          dV -= joint.normalLimiter.normalProjector2 * body2->velocity;
          dV -= joint.normalLimiter.angularProjector2 * body2->angularVelocity;
          dV += joint.normalLimiter.dstVelocity;

          float deltaImpulse = dV * joint.normalLimiter.compInvMass;

          if (deltaImpulse + joint.normalLimiter.accumulatedImpulse < 0.0f)
            deltaImpulse = -joint.normalLimiter.accumulatedImpulse;

          body1->velocity += joint.normalLimiter.compMass1_linear * deltaImpulse;
          body1->angularVelocity += joint.normalLimiter.compMass1_angular * deltaImpulse;
          body2->velocity += joint.normalLimiter.compMass2_linear * deltaImpulse;
          body2->angularVelocity += joint.normalLimiter.compMass2_angular * deltaImpulse;

          joint.normalLimiter.accumulatedImpulse += deltaImpulse;
        }

        float deltaImpulse;

        {
          float dV = 0;

          dV -= joint.frictionLimiter.normalProjector1 * body1->velocity;
          dV -= joint.frictionLimiter.angularProjector1 * body1->angularVelocity;
          dV -= joint.frictionLimiter.normalProjector2 * body2->velocity;
          dV -= joint.frictionLimiter.angularProjector2 * body2->angularVelocity;

          deltaImpulse = dV * joint.frictionLimiter.compInvMass;
        }

        float reactionForce = joint.normalLimiter.accumulatedImpulse;
        float accumulatedImpulse = joint.frictionLimiter.accumulatedImpulse;

        float frictionForce = accumulatedImpulse + deltaImpulse;
        float frictionCoefficient = 0.3f;

        if (fabsf(frictionForce) > (reactionForce * frictionCoefficient))
        {
          float dir = frictionForce > 0.0f ? 1.0f : -1.0f;
          frictionForce = dir * reactionForce * frictionCoefficient;
          deltaImpulse = frictionForce - accumulatedImpulse;
        }

        joint.frictionLimiter.accumulatedImpulse += deltaImpulse;

        body1->velocity += joint.frictionLimiter.compMass1_linear * deltaImpulse;
        body1->angularVelocity += joint.frictionLimiter.compMass1_angular * deltaImpulse;
        
        body2->velocity += joint.frictionLimiter.compMass2_linear * deltaImpulse;
        body2->angularVelocity += joint.frictionLimiter.compMass2_angular * deltaImpulse;
      }
  }

  NOINLINE void SolveJointsDisplacement()
  {
    for (size_t jointIndex = 0; jointIndex < contactJoints.size(); jointIndex++)
    {
      ContactJoint& joint = contactJoints[jointIndex];

      SolveBody* body1 = &solveBodies[joint.body1Index];
      SolveBody* body2 = &solveBodies[joint.body2Index];

      float dV = 0;
      dV -= joint.normalLimiter.normalProjector1 * body1->displacingVelocity;
      dV -= joint.normalLimiter.angularProjector1 * body1->displacingAngularVelocity;
      dV -= joint.normalLimiter.normalProjector2 * body2->displacingVelocity;
      dV -= joint.normalLimiter.angularProjector2 * body2->displacingAngularVelocity;
      dV += joint.normalLimiter.dstDisplacingVelocity;

      float deltaDisplacingImpulse = dV * joint.normalLimiter.compInvMass;

      if (deltaDisplacingImpulse + joint.normalLimiter.accumulatedDisplacingImpulse < 0.0f)
        deltaDisplacingImpulse = -joint.normalLimiter.accumulatedDisplacingImpulse;

      body1->displacingVelocity += joint.normalLimiter.compMass1_linear * deltaDisplacingImpulse;
      body1->displacingAngularVelocity += joint.normalLimiter.compMass1_angular * deltaDisplacingImpulse;
      body2->displacingVelocity += joint.normalLimiter.compMass2_linear * deltaDisplacingImpulse;
      body2->displacingAngularVelocity += joint.normalLimiter.compMass2_angular * deltaDisplacingImpulse;

      joint.normalLimiter.accumulatedDisplacingImpulse += deltaDisplacingImpulse;
    }
  }

  struct SolveBody
  {
     Vector2f velocity;
     float angularVelocity;

     Vector2f displacingVelocity;
     float displacingAngularVelocity;
  };

  AlignedArray<SolveBody> solveBodies;

  std::vector<ContactJoint> contactJoints;

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