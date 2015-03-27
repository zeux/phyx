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

#include <unordered_map>

namespace std
{
  template <> struct hash<std::pair<RigidBody*, RigidBody*>>
  {
    size_t operator()(const std::pair<RigidBody*, RigidBody*>& p) const
    {
      uintptr_t lb = reinterpret_cast<uintptr_t>(p.first);
      uintptr_t rb = reinterpret_cast<uintptr_t>(p.second);

      return lb ^ (rb + 0x9e3779b9 + (lb<<6) + (lb>>2));
    }
  };
}

struct Collider
{
  NOINLINE void FindCollisions(RigidBody *bodies, size_t bodiesCount)
  {
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
          bool& valid = manifoldMap[std::make_pair(&body1, &body2)];

          if (!valid)
          {
            valid = true;
            manifolds.push_back(Manifold(&body1, &body2));
          }
        }
      }
    }
  }

  NOINLINE void UpdateManifolds()
  {
    for (size_t manifoldIndex = 0; manifoldIndex < manifolds.size(); )
    {
      Manifold& m = manifolds[manifoldIndex];

      m.Update();

      if (m.collisionsCount == 0)
      {
        manifoldMap.erase(std::make_pair(m.body1, m.body2));

        if (manifoldIndex != manifolds.size() - 1)
        {
          m = manifolds.back();
          manifolds.pop_back();

          manifoldMap[std::make_pair(m.body1, m.body2)] = manifoldIndex;
        }
        else
        {
          manifolds.pop_back();
        }
      }
      else
      {
        ++manifoldIndex;
      }
    }
  }
  
  std::unordered_map<std::pair<RigidBody*, RigidBody*>, bool> manifoldMap;
  std::vector<Manifold> manifolds;

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