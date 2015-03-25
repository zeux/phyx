#include "Joints.h"
#include <assert.h>
#include <vector>

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
  void SolveJoints(int contactIterationsCount, int penetrationIterationsCount)
  {
    for (int iterationIndex = 0; iterationIndex < contactIterationsCount; iterationIndex++)
    {
      SolveJointsImpulses();
    }
    for (int iterationIndex = 0; iterationIndex < penetrationIterationsCount; iterationIndex++)
    {
      for (size_t jointIndex = 0; jointIndex < contactJoints.size(); jointIndex++)
      {
        contactJoints[jointIndex].SolveDisplacement();
      }
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

  void SolveJointsImpulses()
  {
  #if 1
  SolveJointsImpulsesBaseline();
  #else
  SolveJointsImpulsesExplicit();
  #endif
  }

  void SolveJointsImpulsesBaseline()
  {
    for (size_t jointIndex = 0; jointIndex < contactJoints.size(); jointIndex++)
    {
      contactJoints[jointIndex].SolveImpulse();
    }
  }

  void SolveJointsImpulsesExplicit()
  {
      for (size_t jointIndex = 0; jointIndex < contactJoints.size(); jointIndex++)
      {
        ContactJoint& joint = contactJoints[jointIndex];

        RigidBody* body1 = joint.body1;
        RigidBody* body2 = joint.body2;

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

        if (fabs(frictionForce) > (reactionForce * frictionCoefficient))
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

  std::vector<ContactJoint> contactJoints;
};