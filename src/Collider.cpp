#include "Collider.h"

#include "Parallel.h"
#include "RadixSort.h"

#include "microprofile.h"

Collider::Collider()
    : manifoldMap(std::make_pair(~0u, 0), std::make_pair(~0u, 1))
{
}

NOINLINE void Collider::UpdateBroadphase(RigidBody* bodies, size_t bodiesCount)
{
    MICROPROFILE_SCOPEI("Physics", "UpdateBroadphase", -1);

    broadphase.resize(bodiesCount);
    broadphaseSort[0].resize(bodiesCount);
    broadphaseSort[1].resize(bodiesCount);

    for (size_t bodyIndex = 0; bodyIndex < bodiesCount; ++bodyIndex)
    {
        const AABB2f& aabb = bodies[bodyIndex].geom.aabb;

        broadphaseSort[0][bodyIndex].value = RadixFloatPredicate()(aabb.boxPoint1.x);
        broadphaseSort[0][bodyIndex].index = bodyIndex;
    }

    radixSort3(broadphaseSort[0].data(), broadphaseSort[1].data(), bodiesCount, [](const BroadphaseSortEntry& e) { return e.value; });

    for (size_t i = 0; i < bodiesCount; ++i)
    {
        unsigned int bodyIndex = broadphaseSort[1][i].index;

        const AABB2f& aabb = bodies[bodyIndex].geom.aabb;

        BroadphaseEntry e =
            {
             aabb.boxPoint1.x, aabb.boxPoint2.x,
             (aabb.boxPoint1.y + aabb.boxPoint2.y) * 0.5f,
             (aabb.boxPoint2.y - aabb.boxPoint1.y) * 0.5f,
             unsigned(bodyIndex)};

        broadphase[i] = e;
    }
}

NOINLINE void Collider::UpdatePairs(WorkQueue& queue, RigidBody* bodies, size_t bodiesCount)
{
    assert(bodiesCount == broadphase.size());

    if (queue.getWorkerCount() == 1)
        UpdatePairsSerial(bodies, bodiesCount);
    else
        UpdatePairsParallel(queue, bodies, bodiesCount);
}

NOINLINE void Collider::UpdatePairsSerial(RigidBody* bodies, size_t bodiesCount)
{
    MICROPROFILE_SCOPEI("Physics", "UpdatePairsSerial", -1);

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

NOINLINE void Collider::UpdatePairsParallel(WorkQueue& queue, RigidBody* bodies, size_t bodiesCount)
{
    MICROPROFILE_SCOPEI("Physics", "UpdatePairsParallel", -1);

    manifoldBuffers.resize(queue.getWorkerCount());

    for (auto& buf : manifoldBuffers)
        buf.pairs.clear();

    ParallelFor(queue, bodies, bodiesCount, 128, [this, bodies, bodiesCount](RigidBody& body, int worker) {
        size_t bodyIndex1 = &body - bodies;

        UpdatePairsOne(bodies, bodyIndex1, bodyIndex1 + 1, bodiesCount, manifoldBuffers[worker]);
    });

    MICROPROFILE_SCOPEI("Physics", "CreateManifolds", -1);

    for (auto& buf : manifoldBuffers)
    {
        for (auto& pair : buf.pairs)
        {
            manifoldMap.insert(pair);
            manifolds.push_back(Manifold(&bodies[pair.first], &bodies[pair.second]));
        }
    }
}

void Collider::UpdatePairsOne(RigidBody* bodies, size_t bodyIndex1, size_t startIndex, size_t endIndex, ManifoldDeferredBuffer& buffer)
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

NOINLINE void Collider::UpdateManifolds(WorkQueue& queue)
{
    MICROPROFILE_SCOPEI("Physics", "UpdateManifolds", -1);

    ParallelFor(queue, manifolds.data(), manifolds.size(), 16, [](Manifold& m, int) {
        m.Update();
    });
}

NOINLINE void Collider::PackManifolds()
{
    MICROPROFILE_SCOPEI("Physics", "PackManifolds", -1);

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