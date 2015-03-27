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
  NOINLINE void FindCollisions(RigidBody *bodies, size_t bodiesCount)
  {
    for (ManifoldMap::iterator man = manifolds.begin(); man != manifolds.end(); man++)
      man->second.isMerged = 0;

    UpdateBroadphase(bodies, bodiesCount);
    UpdatePairs(bodies, bodiesCount);
    UpdateManifolds();
  }

  NOINLINE void UpdateBroadphase(RigidBody* bodies, size_t bodiesCount)
  {
    broadphase.clear();

    for (size_t bodyIndex = 0; bodyIndex < bodiesCount; ++bodyIndex)
    {
      BroadphaseEntry e = { bodies[bodyIndex].geom.aabb.boxPoint1.x, unsigned(bodyIndex) };

      broadphase.push_back(e);
    }

    std::sort(broadphase.begin(), broadphase.end());
  }

  NOINLINE void UpdatePairs(RigidBody* bodies, size_t bodiesCount)
  {
    assert(bodiesCount == broadphase.size());

    for (size_t bodyIndex1 = 0; bodyIndex1 < bodiesCount; bodyIndex1++)
    {
      RigidBody& body1 = bodies[broadphase[bodyIndex1].index];
      float maxx = body1.geom.aabb.boxPoint2.x;

      for (size_t bodyIndex2 = bodyIndex1 + 1; bodyIndex2 < bodiesCount; bodyIndex2++)
      {
        if (broadphase[bodyIndex2].minx > maxx)
          break;

        RigidBody& body2 = bodies[broadphase[bodyIndex2].index];

        if (body1.geom.aabb.boxPoint1.y <= body2.geom.aabb.boxPoint2.y && body1.geom.aabb.boxPoint2.y >= body2.geom.aabb.boxPoint1.y)
        {
          BodyPair pair(&body1, &body2);

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
  }

  NOINLINE void UpdateManifolds()
  {
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

  struct BroadphaseEntry
  {
    float minx;
    unsigned int index;

    bool operator<(const BroadphaseEntry& other) const
    {
      return minx < other.minx;
    }
  };
  std::vector<BroadphaseEntry> broadphase;
};