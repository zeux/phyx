#include "Joints.h"
#include <assert.h>
#include <vector>

#include <emmintrin.h>

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

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

  std::vector<SolveBody> solveBodies;

  std::vector<ContactJoint> contactJoints;
};