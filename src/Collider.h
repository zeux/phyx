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

#include "Parallel.h"

namespace std
{
template <>
struct hash<std::pair<unsigned int, unsigned int>>
{
    size_t operator()(const std::pair<unsigned int, unsigned int>& p) const
    {
        unsigned int lb = p.first;
        unsigned int rb = p.second;

        return lb ^ (rb + 0x9e3779b9 + (lb << 6) + (lb >> 2));
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
            const AABB2f& aabb = bodies[bodyIndex].geom.aabb;

            BroadphaseEntry e =
                {
                 aabb.boxPoint1.y, aabb.boxPoint2.y,
                 (aabb.boxPoint1.x + aabb.boxPoint2.x) * 0.5f,
                 (aabb.boxPoint2.x - aabb.boxPoint1.x) * 0.5f,
                 unsigned(bodyIndex)};

            broadphase.push_back(e);
        }

        std::sort(broadphase.begin(), broadphase.end());
    }

    NOINLINE void UpdatePairs(WorkQueue& queue, RigidBody* bodies, size_t bodiesCount)
    {
        assert(bodiesCount == broadphase.size());

        if (queue.getWorkerCount() == 1)
            UpdatePairsSerial(bodies, bodiesCount);
        else
            UpdatePairsParallel(queue, bodies, bodiesCount);
    }

    NOINLINE void UpdatePairsSerial(RigidBody* bodies, size_t bodiesCount)
    {
        for (size_t bodyIndex1 = 0; bodyIndex1 < bodiesCount; bodyIndex1++)
        {
            const BroadphaseEntry& be1 = broadphase[bodyIndex1];
            float maxx = be1.maxx;

            for (size_t bodyIndex2 = bodyIndex1 + 1; bodyIndex2 < bodiesCount; bodyIndex2++)
            {
                const BroadphaseEntry& be2 = broadphase[bodyIndex2];
                if (be2.minx > maxx)
                    break;

                if (fabsf(be2.centery - be1.centery) <= be1.extenty + be2.extenty)
                {
                    if (manifoldMap.insert(std::make_pair(be1.index, be2.index)))
                        manifolds.push_back(Manifold(&bodies[be1.index], &bodies[be2.index]));
                }
            }
        }
    }

    NOINLINE void UpdatePairsParallel(WorkQueue& queue, RigidBody* bodies, size_t bodiesCount)
    {
        manifoldBuffers.resize(queue.getWorkerCount());

        for (auto& buf : manifoldBuffers)
            buf.pairs.clear();

        ParallelFor(queue, bodies, bodiesCount, 128, [this, bodies, bodiesCount](RigidBody& body, int worker) {
            size_t bodyIndex1 = &body - bodies;

            UpdatePairsOne(bodies, bodyIndex1, bodyIndex1 + 1, bodiesCount, manifoldBuffers[worker]);
        });

        for (auto& buf : manifoldBuffers)
        {
            for (auto& pair : buf.pairs)
            {
                manifoldMap.insert(pair);
                manifolds.push_back(Manifold(&bodies[pair.first], &bodies[pair.second]));
            }
        }
    }

    struct ManifoldDeferredBuffer
    {
        std::vector<std::pair<int, int>> pairs;
    };

    void UpdatePairsOne(RigidBody* bodies, size_t bodyIndex1, size_t startIndex, size_t endIndex, ManifoldDeferredBuffer& buffer)
    {
        const BroadphaseEntry& be1 = broadphase[bodyIndex1];
        float maxx = be1.maxx;

        for (size_t bodyIndex2 = startIndex; bodyIndex2 < endIndex; bodyIndex2++)
        {
            const BroadphaseEntry& be2 = broadphase[bodyIndex2];
            if (be2.minx > maxx)
                return;

            if (fabsf(be2.centery - be1.centery) <= be1.extenty + be2.extenty)
            {
                if (!manifoldMap.contains(std::make_pair(be1.index, be2.index)))
                    buffer.pairs.push_back(std::make_pair(be1.index, be2.index));
            }
        }
    }

    NOINLINE void UpdateManifolds(WorkQueue& queue)
    {
        ParallelFor(queue, manifolds.data(), manifolds.size(), 16, [](Manifold& m, int) {
            m.Update();
        });
    }

    NOINLINE void PackManifolds()
    {
        for (size_t manifoldIndex = 0; manifoldIndex < manifolds.size();)
        {
            Manifold& m = manifolds[manifoldIndex];

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

    DenseHashSet<std::pair<unsigned int, unsigned int>> manifoldMap;
    std::vector<Manifold> manifolds;
    std::vector<ManifoldDeferredBuffer> manifoldBuffers;

    struct BroadphaseEntry
    {
        float minx, maxx;
        float centery, extenty;
        unsigned int index;

        bool operator<(const BroadphaseEntry& other) const
        {
            return minx < other.minx;
        }
    };

    std::vector<BroadphaseEntry> broadphase;
};