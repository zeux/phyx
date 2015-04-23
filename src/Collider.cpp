#include "Collider.h"

#include "base/Parallel.h"
#include "base/RadixSort.h"

#include "microprofile.h"

static NOINLINE bool ComputeSeparatingAxis(RigidBody* body1, RigidBody* body2, Vector2f& separatingAxis)
{
    // http://www.geometrictools.com/Source/Intersection2D.html#PlanarPlanar
    // Adapted to return axis with least amount of penetration
    const Vector2f* A0 = &body1->coords.xVector;
    const Vector2f* A1 = &body2->coords.xVector;
    const Vector2f& E0 = body1->geom.size;
    const Vector2f& E1 = body2->geom.size;

    Vector2f D = body1->coords.pos - body2->coords.pos;

    float Adot[2][2];

    // Test axis box0.axis[0]
    Adot[0][0] = fabsf(A0[0] * A1[0]);
    Adot[0][1] = fabsf(A0[0] * A1[1]);

    float rSum0 = E0.x + E1.x * Adot[0][0] + E1.y * Adot[0][1];
    float dist0 = fabsf(A0[0] * D) - rSum0;
    if (dist0 > 0) return false;

    float bestdist = dist0;
    Vector2f bestaxis = A0[0];

    // Test axis box0.axis[1].
    Adot[1][0] = fabsf(A0[1] * A1[0]);
    Adot[1][1] = fabsf(A0[1] * A1[1]);

    float rSum1 = E0.y + E1.x * Adot[1][0] + E1.y * Adot[1][1];
    float dist1 = fabsf(A0[1] * D) - rSum1;
    if (dist1 > 0) return false;
    if (dist1 > bestdist) { bestdist = dist1; bestaxis = A0[1]; }

    // Test axis box1.axis[0].
    float rSum2 = E1.x + E0.x * Adot[0][0] + E0.y * Adot[1][0];
    float dist2 = fabsf(A1[0] * D) - rSum2;
    if (dist2 > 0) return false;
    if (dist2 > bestdist) { bestdist = dist2; bestaxis = A1[0]; }

    // Test axis box1.axis[1].
    float rSum3 = E1.y + E0.x * Adot[0][1] + E0.y * Adot[1][1];
    float dist3 = fabsf(A1[1] * D) - rSum3;
    if (dist3 > 0) return false;
    if (dist3 > bestdist) { bestdist = dist3; bestaxis = A1[1]; }

    separatingAxis = bestaxis;
    return true;
}

static void AddPoint(ContactPoint* points, int& pointCount, ContactPoint& newbie)
{
    ContactPoint* closest = 0;
    float bestdepth = std::numeric_limits<float>::max();

    for (int collisionIndex = 0; collisionIndex < pointCount; collisionIndex++)
    {
        ContactPoint& col = points[collisionIndex];

        if (newbie.Equals(col, 2.0f))
        {
            float depth = (newbie.delta1 - col.delta1).SquareLen() + (newbie.delta2 - col.delta2).SquareLen();
            if (depth < bestdepth)
            {
                bestdepth = depth;
                closest = &col;
            }
        }
    }

    if (closest)
    {
        closest->isMerged = 1;
        closest->isNewlyCreated = 0;
        closest->normal = newbie.normal;
        closest->delta1 = newbie.delta1;
        closest->delta2 = newbie.delta2;
    }
    else
    {
        assert(pointCount < 4);
        newbie.isMerged = 1;
        newbie.isNewlyCreated = 1;
        points[pointCount++] = newbie;
    }
}

static void NOINLINE GenerateContacts(RigidBody* body1, RigidBody* body2, ContactPoint* points, int& pointCount, Vector2f separatingAxis)
{
    if (separatingAxis * (body1->coords.pos - body2->coords.pos) < 0.0f)
        separatingAxis.Invert();

    const int kMaxSupportPoints = 2;
    Vector2f supportPoints1[kMaxSupportPoints];
    Vector2f supportPoints2[kMaxSupportPoints];

    float linearTolerance = 2.0f;

    int supportPointsCount1 = body1->geom.GetSupportPointSet(-separatingAxis, supportPoints1);
    int supportPointsCount2 = body2->geom.GetSupportPointSet(separatingAxis, supportPoints2);

    if ((supportPointsCount1 == 2) && (((supportPoints1[0] - supportPoints1[1])).SquareLen() < linearTolerance * linearTolerance))
    {
        supportPoints1[0] = (supportPoints1[0] + supportPoints1[1]) * 0.5f;
        supportPointsCount1 = 1;
    }
    if ((supportPointsCount2 == 2) && (((supportPoints2[0] - supportPoints2[1])).SquareLen() < linearTolerance * linearTolerance))
    {
        supportPoints2[0] = (supportPoints2[0] + supportPoints2[1]) * 0.5f;
        supportPointsCount2 = 1;
    }

    if ((supportPointsCount1 == 1) && (supportPointsCount2 == 1))
    {
        Vector2f delta = supportPoints2[0] - supportPoints1[0];
        //float eps = (delta ^ separatingAxis).SquareLen();
        if (delta * separatingAxis >= 0.0f)
        {
            ContactPoint newbie(supportPoints1[0], supportPoints2[0], separatingAxis, body1, body2);
            AddPoint(points, pointCount, newbie);
        }
    }
    else if ((supportPointsCount1 == 1) && (supportPointsCount2 == 2))
    {
        Vector2f n = (supportPoints2[1] - supportPoints2[0]).GetPerpendicular();
        Vector2f point;
        ProjectPointToLine(supportPoints1[0], supportPoints2[0], n, separatingAxis, point);

        if ((((point - supportPoints2[0]) * (supportPoints2[1] - supportPoints2[0])) >= 0.0f) &&
            (((point - supportPoints2[1]) * (supportPoints2[0] - supportPoints2[1])) >= 0.0f))
        {
            ContactPoint newbie(supportPoints1[0], point, separatingAxis, body1, body2);
            AddPoint(points, pointCount, newbie);
        }
    }
    else if ((supportPointsCount1 == 2) && (supportPointsCount2 == 1))
    {
        Vector2f n = (supportPoints1[1] - supportPoints1[0]).GetPerpendicular();
        Vector2f point;
        ProjectPointToLine(supportPoints2[0], supportPoints1[0], n, separatingAxis, point);

        if ((((point - supportPoints1[0]) * (supportPoints1[1] - supportPoints1[0])) >= 0.0f) &&
            (((point - supportPoints1[1]) * (supportPoints1[0] - supportPoints1[1])) >= 0.0f))
        {
            ContactPoint newbie(point, supportPoints2[0], separatingAxis, body1, body2);
            AddPoint(points, pointCount, newbie);
        }
    }
    else if ((supportPointsCount2 == 2) && (supportPointsCount1 == 2))
    {
        struct TempColInfo
        {
            Vector2f point1, point2;
        };
        TempColInfo tempCol[4];
        int tempCols = 0;
        for (int i = 0; i < 2; i++)
        {
            Vector2f n = (supportPoints2[1] - supportPoints2[0]).GetPerpendicular();
            if ((supportPoints1[i] - supportPoints2[0]) * n >= 0.0)
            {
                Vector2f point;
                ProjectPointToLine(supportPoints1[i], supportPoints2[0], n, separatingAxis, point);

                if ((((point - supportPoints2[0]) * (supportPoints2[1] - supportPoints2[0])) >= 0.0f) &&
                    (((point - supportPoints2[1]) * (supportPoints2[0] - supportPoints2[1])) >= 0.0f))
                {
                    tempCol[tempCols].point1 = supportPoints1[i];
                    tempCol[tempCols].point2 = point;
                    tempCols++;
                }
            }
        }
        for (int i = 0; i < 2; i++)
        {
            Vector2f n = (supportPoints1[1] - supportPoints1[0]).GetPerpendicular();
            if ((supportPoints2[i] - supportPoints1[0]) * n >= 0.0)
            {
                Vector2f point;
                ProjectPointToLine(supportPoints2[i], supportPoints1[0], n, separatingAxis, point);

                if ((((point - supportPoints1[0]) * (supportPoints1[1] - supportPoints1[0])) >= 0.0f) &&
                    (((point - supportPoints1[1]) * (supportPoints1[0] - supportPoints1[1])) >= 0.0f))
                {
                    tempCol[tempCols].point1 = point;
                    tempCol[tempCols].point2 = supportPoints2[i];
                    tempCols++;
                }
            }
        }

        if (tempCols == 1) //buggy but must work
        {
            ContactPoint newbie(tempCol[0].point1, tempCol[0].point2, separatingAxis, body1, body2);
            AddPoint(points, pointCount, newbie);
        }
        if (tempCols >= 2) //means only equality, but clamp to two points
        {
            ContactPoint newbie1(tempCol[0].point1, tempCol[0].point2, separatingAxis, body1, body2);
            AddPoint(points, pointCount, newbie1);
            ContactPoint newbie2(tempCol[1].point1, tempCol[1].point2, separatingAxis, body1, body2);
            AddPoint(points, pointCount, newbie2);
        }
    }
}

static void UpdateManifold(Manifold& m, RigidBody* bodies, ContactPoint* points)
{
    ContactPoint newpoints[kMaxContactPoints * 2];

    for (int collisionIndex = 0; collisionIndex < m.pointCount; collisionIndex++)
    {
        newpoints[collisionIndex] = points[collisionIndex];
        newpoints[collisionIndex].isMerged = 0;
        newpoints[collisionIndex].isNewlyCreated = 0;
    }

    int newPointCount = m.pointCount;

    RigidBody* body1 = &bodies[m.body1Index];
    RigidBody* body2 = &bodies[m.body2Index];

    Vector2f separatingAxis;
    if (ComputeSeparatingAxis(body1, body2, separatingAxis))
    {
        GenerateContacts(body1, body2, newpoints, newPointCount, separatingAxis);
    }

    m.pointCount = 0;

    for (int collisionIndex = 0; collisionIndex < newPointCount; ++collisionIndex)
    {
        if (newpoints[collisionIndex].isMerged)
        {
            assert(m.pointCount < kMaxContactPoints);
            points[m.pointCount++] = newpoints[collisionIndex];
        }
    }
}

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

        broadphaseSort[0][bodyIndex].value = radixFloat(aabb.boxPoint1.x);
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
                {
                    manifolds.push_back(Manifold(be1.index, be2.index, manifolds.size() * 2));
                }
            }
        }
    }
}

NOINLINE void Collider::UpdatePairsParallel(WorkQueue& queue, RigidBody* bodies, size_t bodiesCount)
{
    MICROPROFILE_SCOPEI("Physics", "UpdatePairsParallel", -1);

    manifoldBuffers.resize(queue.getWorkerCount() + 1);

    for (auto& buf : manifoldBuffers)
        buf.pairs.clear();

    parallelFor(queue, 0, bodiesCount, 128, [this, bodies, bodiesCount](int bodyIndex1, int worker) {
        UpdatePairsOne(bodies, bodyIndex1, bodyIndex1 + 1, bodiesCount, manifoldBuffers[worker]);
    });

    MICROPROFILE_SCOPEI("Physics", "CreateManifolds", -1);

    for (auto& buf : manifoldBuffers)
    {
        for (auto& pair : buf.pairs)
        {
            manifoldMap.insert(pair);
            manifolds.push_back(Manifold(pair.first, pair.second, manifolds.size() * kMaxContactPoints));
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
            {
                buffer.pairs.push_back(std::make_pair(be1.index, be2.index));
            }
        }
    }
}

NOINLINE void Collider::UpdateManifolds(WorkQueue& queue, RigidBody* bodies)
{
    MICROPROFILE_SCOPEI("Physics", "UpdateManifolds", -1);

    contactPoints.resize(manifolds.size() * kMaxContactPoints);

    parallelFor(queue, manifolds.data(), manifolds.size(), 16, [&](Manifold& m, int) {
        UpdateManifold(m, bodies, contactPoints.data + m.pointIndex);
    });
}

NOINLINE void Collider::PackManifolds(RigidBody* bodies)
{
    MICROPROFILE_SCOPEI("Physics", "PackManifolds", -1);

    for (size_t manifoldIndex = 0; manifoldIndex < manifolds.size();)
    {
        Manifold& m = manifolds[manifoldIndex];

        // TODO
        // This reduces broadphase insert/erase operations, which is good
        // However, current behavior causes issues with DenseHash - is it possible to improve it?
        if (m.pointCount == 0 && !bodies[m.body1Index].geom.aabb.Intersects(bodies[m.body2Index].geom.aabb))
        {
            manifoldMap.erase(std::make_pair(m.body1Index, m.body2Index));

            if (manifoldIndex < manifolds.size())
            {
                Manifold& me = manifolds.back();

                unsigned int pointIndex = m.pointIndex;

                for (int i = 0; i < me.pointCount; ++i)
                    contactPoints[pointIndex + i] = contactPoints[me.pointIndex + i];

                m = me;
                m.pointIndex = pointIndex;
            }

            manifolds.pop_back();
        }
        else
        {
            ++manifoldIndex;
        }
    }

    contactPoints.resize(manifolds.size() * kMaxContactPoints);
}