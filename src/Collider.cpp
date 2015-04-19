#include "Collider.h"

#include "Parallel.h"
#include "RadixSort.h"

#include "microprofile.h"

static bool ComputeSeparatingAxis(RigidBody* body1, RigidBody* body2, Vector2f& separatingAxis)
{
    Vector2f axis[4];
    axis[0] = body1->coords.xVector;
    axis[1] = body1->coords.yVector;
    axis[2] = body2->coords.xVector;
    axis[3] = body2->coords.yVector;

    bool found = 0;
    float bestquaddepth = 1e5f;
    Vector2f bestaxis;

    for (int i = 0; i < 4; i++)
    {
        float min1, max1;
        float min2, max2;
        body1->geom.GetAxisProjectionRange(axis[i], min1, max1);
        body2->geom.GetAxisProjectionRange(axis[i], min2, max2);
        if ((min1 > max2) || (min2 > max1))
        {
            return 0;
        }

        float delta = std::min(max2 - min1, max1 - min2);
        if (bestquaddepth > delta)
        {
            bestquaddepth = delta;
            bestaxis = axis[i];
            found = 1;
        }
    }
    separatingAxis = bestaxis;
    return found;
}

static bool ComputeSeparatingAxis_SSE2(RigidBody* body1, RigidBody* body2, Vector2f& separatingAxis)
{
    const Vector2f* A0 = &body1->coords.xVector;
    const Vector2f* A1 = &body2->coords.xVector;
    const Vector2f& E0 = body1->geom.size;
    const Vector2f& E1 = body2->geom.size;

    Vector2f D = body1->coords.pos - body2->coords.pos;

    float absA0dA1[2][2], rSum, pd;

    float bestpd;
    Vector2f bestaxis;

    // Test axis box0.axis[0].
    absA0dA1[0][0] = std::abs(A0[0] * A1[0]);
    absA0dA1[0][1] = std::abs(A0[0] * A1[1]);
    rSum = E0.x + E1.x * absA0dA1[0][0] + E1.y * absA0dA1[0][1];
    pd = std::abs(A0[0] * D) - rSum;
    if (pd > 0) return false;

    bestpd = pd;
    bestaxis = A0[0];

    // Test axis box0.axis[1].
    absA0dA1[1][0] = std::abs(A0[1] * A1[0]);
    absA0dA1[1][1] = std::abs(A0[1] * A1[1]);
    rSum = E0.y + E1.x * absA0dA1[1][0] + E1.y * absA0dA1[1][1];
    pd = std::abs(A0[1] * D) - rSum;
    if (pd > 0) return false;

    if (pd > bestpd)
    {
        bestpd = pd;
        bestaxis = A0[1];
    }

    // Test axis box1.axis[0].
    rSum = E1.x + E0.x * absA0dA1[0][0] + E0.y * absA0dA1[1][0];
    pd = std::abs(A1[0] * D) - rSum;
    if (pd > 0) return false;

    if (pd > bestpd)
    {
        bestpd = pd;
        bestaxis = A1[0];
    }

    // Test axis box1.axis[1].
    rSum = E1.y + E0.x * absA0dA1[0][1] + E0.y * absA0dA1[1][1];
    pd = std::abs(A1[1] * D) - rSum;
    if (pd > 0) return false;

    if (pd > bestpd)
    {
        bestpd = pd;
        bestaxis = A1[1];
    }

    separatingAxis = bestaxis;
    return true;
}

static void AddPoint(Manifold& m, ContactPoint& newbie)
{
    ContactPoint* closest = 0;
    float bestdepth = std::numeric_limits<float>::max();

    for (int collisionIndex = 0; collisionIndex < m.pointCount; collisionIndex++)
    {
        ContactPoint& col = m.points[collisionIndex];

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
        assert(collisionsCount < 4);
        newbie.isMerged = 1;
        newbie.isNewlyCreated = 1;
        m.points[m.pointCount++] = newbie;
    }
}

static void GenerateContacts(Manifold& m, Vector2f separatingAxis)
{
    RigidBody* body1 = m.body1;
    RigidBody* body2 = m.body2;

    if (separatingAxis * (body1->coords.pos - body2->coords.pos) < 0.0f)
        separatingAxis.Invert();

    const int maxSupportPoints = 2;
    Vector2f supportPoints1[maxSupportPoints];
    Vector2f supportPoints2[maxSupportPoints];

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
        if (delta * separatingAxis > 0.0f)
        {
            ContactPoint newbie(supportPoints1[0], supportPoints2[0], separatingAxis, body1, body2);
            AddPoint(m, newbie);
        }
    }
    else if ((supportPointsCount1 == 1) && (supportPointsCount2 == 2))
    {
        Vector2f n = (supportPoints2[1] - supportPoints2[0]).GetPerpendicular();
        Vector2f point;
        ProjectPointToLine(supportPoints1[0], supportPoints2[0], n, separatingAxis, point);

        if ((((point - supportPoints2[0]) * (supportPoints2[1] - supportPoints2[0])) > 0.0f) &&
            (((point - supportPoints2[1]) * (supportPoints2[0] - supportPoints2[1])) > 0.0f))
        {
            ContactPoint newbie(supportPoints1[0], point, separatingAxis, body1, body2);
            AddPoint(m, newbie);
        }
    }
    else if ((supportPointsCount1 == 2) && (supportPointsCount2 == 1))
    {
        Vector2f n = (supportPoints1[1] - supportPoints1[0]).GetPerpendicular();
        Vector2f point;
        ProjectPointToLine(supportPoints2[0], supportPoints1[0], n, separatingAxis, point);

        if ((((point - supportPoints1[0]) * (supportPoints1[1] - supportPoints1[0])) > 0.0f) &&
            (((point - supportPoints1[1]) * (supportPoints1[0] - supportPoints1[1])) > 0.0f))
        {
            ContactPoint newbie(point, supportPoints2[0], separatingAxis, body1, body2);
            AddPoint(m, newbie);
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
            if ((supportPoints1[i] - supportPoints2[0]) * n > 0.0)
            {
                Vector2f point;
                ProjectPointToLine(supportPoints1[i], supportPoints2[0], n, separatingAxis, point);

                if ((((point - supportPoints2[0]) * (supportPoints2[1] - supportPoints2[0])) >= 0.0f) &&
                    (((point - supportPoints2[1]) * (supportPoints2[0] - supportPoints2[1])) > 0.0f))
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
            if ((supportPoints2[i] - supportPoints1[0]) * n > 0.0)
            {
                Vector2f point;
                ProjectPointToLine(supportPoints2[i], supportPoints1[0], n, separatingAxis, point);

                if ((((point - supportPoints1[0]) * (supportPoints1[1] - supportPoints1[0])) >= 0.0f) &&
                    (((point - supportPoints1[1]) * (supportPoints1[0] - supportPoints1[1])) > 0.0f))
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
            AddPoint(m, newbie);
        }
        if (tempCols >= 2) //means only equality, but clamp to two points
        {
            ContactPoint newbie1(tempCol[0].point1, tempCol[0].point2, separatingAxis, body1, body2);
            AddPoint(m, newbie1);
            ContactPoint newbie2(tempCol[1].point1, tempCol[1].point2, separatingAxis, body1, body2);
            AddPoint(m, newbie2);
        }
    }
}

static void UpdateManifold(Manifold& m)
{
    for (int collisionIndex = 0; collisionIndex < m.pointCount; collisionIndex++)
    {
        m.points[collisionIndex].isMerged = 0;
        m.points[collisionIndex].isNewlyCreated = 0;
    }
    Vector2f separatingAxis;
    if (ComputeSeparatingAxis(m.body1, m.body2, separatingAxis))
    {
        GenerateContacts(m, separatingAxis);
    }

    for (int collisionIndex = 0; collisionIndex < m.pointCount;)
    {
        if (!m.points[collisionIndex].isMerged)
        {
            m.points[collisionIndex] = m.points[m.pointCount - 1];
            m.pointCount--;
        }
        else
        {
            collisionIndex++;
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
        UpdateManifold(m);
    });
}

NOINLINE void Collider::PackManifolds()
{
    MICROPROFILE_SCOPEI("Physics", "PackManifolds", -1);

    for (size_t manifoldIndex = 0; manifoldIndex < manifolds.size();)
    {
        Manifold& m = manifolds[manifoldIndex];

        if (m.pointCount == 0)
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