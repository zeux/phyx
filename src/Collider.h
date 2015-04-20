#pragma once

#include "Manifold.h"
#include "base/DenseHash.h"
#include "base/AlignedArray.h"

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

class WorkQueue;

struct Collider
{
    Collider();

    void UpdateBroadphase(RigidBody* bodies, size_t bodiesCount);
    void UpdatePairs(WorkQueue& queue, RigidBody* bodies, size_t bodiesCount);
    void UpdatePairsSerial(RigidBody* bodies, size_t bodiesCount);
    void UpdatePairsParallel(WorkQueue& queue, RigidBody* bodies, size_t bodiesCount);

    struct ManifoldDeferredBuffer;

    void UpdatePairsOne(RigidBody* bodies, size_t bodyIndex1, size_t startIndex, size_t endIndex, ManifoldDeferredBuffer& buffer);

    void UpdateManifolds(WorkQueue& queue);
    void PackManifolds();

    struct ManifoldDeferredBuffer
    {
        std::vector<std::pair<int, int>> pairs;
    };

    struct BroadphaseEntry
    {
        float minx, maxx;
        float centery, extenty;
        unsigned int index;
    };

    struct BroadphaseSortEntry
    {
        unsigned int value;
        unsigned int index;
    };

    DenseHashSet<std::pair<unsigned int, unsigned int>> manifoldMap;

    std::vector<Manifold> manifolds;
    AlignedArray<ContactPoint> contactPoints;

    std::vector<ManifoldDeferredBuffer> manifoldBuffers;

    std::vector<BroadphaseEntry> broadphase;
    std::vector<BroadphaseSortEntry> broadphaseSort[2];
};