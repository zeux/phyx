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
  std::vector<ContactJoint> contactJoints;
};