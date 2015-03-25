#pragma once
#include "RigidBody.h"
#include "Collider.h"
#include "Solver.h"
#include <vector>
#include <SFML/Graphics.hpp>

struct PhysSystem
{
  RigidBody *AddBody(Coords2f coords, Vector2f size)
  {
    RigidBody newbie(coords, size, 1e-5f);
    bodies.push_back(newbie);
    return &(bodies[bodies.size() - 1]);
  }
  void Update(float dt)
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
    for (Collider::ManifoldMap::iterator man = collider.manifolds.begin(); man != collider.manifolds.end(); man++)
    {
      for (int collisionIndex = 0; collisionIndex < man->second.collisionsCount; collisionIndex++)
      {
        Collision &col = man->second.collisions[collisionIndex];
        if (!col.userInfo) continue;

        ContactJoint::Descriptor desc;
        desc.body1 = man->second.body1;
        desc.body2 = man->second.body2;
        desc.collision = &col;
        solver.RefreshContactJoint(desc);
      }
    }
    for (Collider::ManifoldMap::iterator man = collider.manifolds.begin(); man != collider.manifolds.end(); man++)
    {
      for (int collisionIndex = 0; collisionIndex < man->second.collisionsCount; collisionIndex++)
      {
        Collision &col = man->second.collisions[collisionIndex];
        if (col.userInfo) continue;
        ContactJoint::Descriptor desc;
        desc.body1 = man->second.body1;
        desc.body2 = man->second.body2;
        desc.collision = &col;
        solver.contactJoints.push_back(ContactJoint(desc));
      }
    }
    solver.RefreshJoints();
    mergeTime += clock.getElapsedTime().asSeconds();
    clock.restart();

    solver.SolveJoints(15, 15);
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