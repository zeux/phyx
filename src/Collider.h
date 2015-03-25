#pragma once
#include "Vector2.h"
#include "Coords2.h"
#include "AABB2.h"
#include "Geom.h"
#include <assert.h>
#include <algorithm>
#include "Collision.h"
#include "Manifold.h"
#include "RigidBody.h"
#include <map>

struct Collider
{
  void FindCollisions(RigidBody *bodies, size_t bodiesCount)
  {
    for (ManifoldMap::iterator man = manifolds.begin(); man != manifolds.end(); man++)
    {
      man->second.isMerged = 0;
    }
    for (size_t bodyIndex1 = 0; bodyIndex1 < bodiesCount; bodyIndex1++)
    {
      for (size_t bodyIndex2 = bodyIndex1 + 1; bodyIndex2 < bodiesCount; bodyIndex2++)
      {
        if (bodies[bodyIndex1].geom.aabb.Intersects(bodies[bodyIndex2].geom.aabb))
        {
          if (bodyIndex1 == 0 && bodyIndex2 == 1)
          {
            int pp = 1;
          }
          BodyPair pair(&bodies[bodyIndex1], &bodies[bodyIndex2]);
          ManifoldMap::iterator man = manifolds.find(pair);
          if (man != manifolds.end())
          {
            man->second.isMerged = true;
          }
          else
          {
            manifolds[pair] = Manifold(pair.body1, pair.body2);
          }
        }
      }
    }
    for (ManifoldMap::iterator man = manifolds.begin(); man != manifolds.end();)
    {
      man->second.Update();
      if (man->second.collisionsCount == 0)
      {
        ManifoldMap::iterator next = man;
        next++;
        manifolds.erase(man);
        man = next;
      }
      else
      {
        man++;
      }
    }
  }
  struct BodyPair
  {
    BodyPair()
    {
      body1 = body2 = 0;
    }
    BodyPair(RigidBody *body1, RigidBody *body2)
    {
      this->body1 = body1;
      this->body2 = body2;
    }
    bool operator < (const BodyPair &other) const
    {
      if (body1 < other.body1) return 1;
      if ((body1 == other.body1) && (body2 < other.body2)) return 1;
      return 0;
    }

    RigidBody *body1;
    RigidBody *body2;
  };

  typedef std::map<BodyPair, Manifold> ManifoldMap;
  ManifoldMap manifolds;
};