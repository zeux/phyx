#pragma once

#include <math.h>

#include "RigidBody.h"
#include "Collider.h"
#include "Solver.h"
#include <vector>
#include <SFML/Graphics.hpp>

struct PhysSystem
{
  enum SolveMode
  {
    Solve_Baseline,
    Solve_AoS,
    Solve_SoA_Scalar,
    Solve_SoA_SSE2,
    Solve_SoA_AVX2,
    Solve_SoAPacked_Scalar,
    Solve_SoAPacked_SSE2,
    Solve_SoAPacked_AVX2,
    Solve_SoAPacked_FMA,
  };

  RigidBody *AddBody(Coords2f coords, Vector2f size)
  {
    RigidBody newbie(coords, size, 1e-5f);
    newbie.index = bodies.size();
    bodies.push_back(newbie);
    return &(bodies[bodies.size() - 1]);
  }
  void Update(float dt, SolveMode mode)
  {
    collisionTime = mergeTime = solveTime = 0;

    sf::Clock clock;
    for (size_t bodyIndex = 0; bodyIndex < bodies.size(); bodyIndex++)
    {
      bodies[bodyIndex].IntegrateVelocity(dt);
    }
    mergeTime += clock.getElapsedTime().asSeconds();
    clock.restart();

    collider.FindCollisions(bodies.data(), bodies.size());
    collisionTime += clock.getElapsedTime().asSeconds();
    clock.restart();

    for (size_t jointIndex = 0; jointIndex < solver.contactJoints.size(); jointIndex++)
    {
      solver.contactJoints[jointIndex].valid = 0;
    }

    for (size_t manifoldIndex = 0; manifoldIndex < collider.manifolds.size(); ++manifoldIndex)
    {
      Manifold& man = collider.manifolds[manifoldIndex];

      for (int collisionIndex = 0; collisionIndex < man.collisionsCount; collisionIndex++)
      {
        Collision &col = man.collisions[collisionIndex];

        ContactJoint::Descriptor desc;
        desc.body1 = man.body1;
        desc.body2 = man.body2;
        desc.collision = &col;

        if (col.solverIndex < 0)
        {
          solver.contactJoints.push_back(ContactJoint(desc, solver.contactJoints.size()));
        }
        else
        {
          solver.RefreshContactJoint(desc, col.solverIndex);
        }
      }
    }

    solver.RefreshJoints();
    mergeTime += clock.getElapsedTime().asSeconds();
    clock.restart();

    int contactIterationsCount = 15;
    int penetrationIterationsCount = 15;

    switch (mode)
    {
    case Solve_AoS:
      solver.SolveJointsAoS(contactIterationsCount, penetrationIterationsCount);
      break;

    case Solve_SoA_Scalar:
      solver.SolveJointsSoA_Scalar(&bodies[0], bodies.size(), contactIterationsCount, penetrationIterationsCount);
      break;

    case Solve_SoA_SSE2:
      solver.SolveJointsSoA_SSE2(&bodies[0], bodies.size(), contactIterationsCount, penetrationIterationsCount);
      break;

  #ifdef __AVX2__
    case Solve_SoA_AVX2:
      solver.SolveJointsSoA_AVX2(&bodies[0], bodies.size(), contactIterationsCount, penetrationIterationsCount);
      break;
  #endif

    case Solve_SoAPacked_Scalar:
      solver.SolveJointsSoAPacked_Scalar(&bodies[0], bodies.size(), contactIterationsCount, penetrationIterationsCount);
      break;

    case Solve_SoAPacked_SSE2:
      solver.SolveJointsSoAPacked_SSE2(&bodies[0], bodies.size(), contactIterationsCount, penetrationIterationsCount);
      break;

  #ifdef __AVX2__
    case Solve_SoAPacked_AVX2:
      solver.SolveJointsSoAPacked_AVX2(&bodies[0], bodies.size(), contactIterationsCount, penetrationIterationsCount);
      break;
  #endif

  #if defined(__AVX2__) && defined(__FMA__)
    case Solve_SoAPacked_FMA:
      solver.SolveJointsSoAPacked_FMA(&bodies[0], bodies.size(), contactIterationsCount, penetrationIterationsCount);
      break;
  #endif

    default:
      solver.SolveJoints(contactIterationsCount, penetrationIterationsCount);
    }

    solveTime += clock.getElapsedTime().asSeconds();
    clock.restart();
    for (size_t bodyIndex = 0; bodyIndex < bodies.size(); bodyIndex++)
    {
      bodies[bodyIndex].IntegratePosition(dt);
    }
    mergeTime += clock.getElapsedTime().asSeconds();
    clock.restart();
  }
  size_t GetBodiesCount()
  {
    return bodies.size();
  }
  RigidBody *GetBody(int index)
  {
    return &(bodies[index]);
  }
  int GetJointsCount()
  {
    return solver.contactJoints.size();
  }
  Collider *GetCollider()
  {
    return &collider;
  }

  float collisionTime;
  float mergeTime;
  float solveTime;
private:
  std::vector<RigidBody> bodies;
  Collider collider;
  Solver solver;
};