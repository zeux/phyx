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

#include "DenseHash.h"

namespace std
{
  template <> struct hash<std::pair<unsigned int, unsigned int>>
  {
    size_t operator()(const std::pair<unsigned int, unsigned int>& p) const
    {
      unsigned int lb = p.first;
      unsigned int rb = p.second;

      return lb ^ (rb + 0x9e3779b9 + (lb<<6) + (lb>>2));
    }
  };
}

struct Collider
{
  Collider()
  : manifoldMap(std::make_pair(~0u, 0), std::make_pair(~0u, 1))
  {
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
      unsigned int rigidBodyIndex1 = broadphase[bodyIndex1].index;
      RigidBody& body1 = bodies[rigidBodyIndex1];
      float maxx = body1.geom.aabb.boxPoint2.x;

      for (size_t bodyIndex2 = bodyIndex1 + 1; bodyIndex2 < bodiesCount; bodyIndex2++)
      {
        if (broadphase[bodyIndex2].minx > maxx)
          break;

        unsigned int rigidBodyIndex2 = broadphase[bodyIndex2].index;
        RigidBody& body2 = bodies[rigidBodyIndex2];

        if (body1.geom.aabb.boxPoint1.y <= body2.geom.aabb.boxPoint2.y && body1.geom.aabb.boxPoint2.y >= body2.geom.aabb.boxPoint1.y)
        {
          if (manifoldMap.insert(std::make_pair(rigidBodyIndex1, rigidBodyIndex2)))
            manifolds.push_back(Manifold(&body1, &body2));
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
        manifoldMap.erase(std::make_pair(m.body1->index, m.body2->index));

        m = manifolds.back();
        manifolds.pop_back();
      }
      else
      {
        ++manifoldIndex;
      }
    }
  }
  
  DenseHashSet<std::pair<unsigned int, unsigned int> > manifoldMap;
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