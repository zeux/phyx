#include "World.h"

#include "base/WorkQueue.h"
#include "microprofile.h"

World::World()
    : gravity(0)
{
}

RigidBody* World::AddBody(Coords2f coords, Vector2f size)
{
    RigidBody newbie(coords, size, 1e-5f);
    newbie.index = bodies.size();
    bodies.push_back(newbie);
    return &(bodies[bodies.size() - 1]);
}

void World::Update(WorkQueue& queue, float dt, SolveMode mode, int contactIterationsCount, int penetrationIterationsCount, bool useIslands)
{
    MICROPROFILE_SCOPEI("Physics", "Update", 0x00ff00);

    collisionTime = mergeTime = solveTime = 0;

    ApplyGravity();
    IntegrateVelocity(dt);

    collider.UpdateBroadphase(bodies.data(), bodies.size());
    collider.UpdatePairs(queue, bodies.data(), bodies.size());
    collider.UpdateManifolds(queue);
    collider.PackManifolds();

    RefreshContactJoints();

    switch (mode)
    {
    case Solve_Scalar:
        solver.SolveJoints_Scalar(queue, bodies.data(), bodies.size(), collider.contactPoints.data, contactIterationsCount, penetrationIterationsCount, useIslands);
        break;

    case Solve_SSE2:
        solver.SolveJoints_SSE2(queue, bodies.data(), bodies.size(), collider.contactPoints.data, contactIterationsCount, penetrationIterationsCount, useIslands);
        break;

#ifdef __AVX2__
    case Solve_AVX2:
        solver.SolveJoints_AVX2(queue, bodies.data(), bodies.size(), collider.contactPoints.data, contactIterationsCount, penetrationIterationsCount, useIslands);
        break;
#endif

    default:
        assert(!"Unknown solver mode");
        break;
    }

    IntegratePosition(dt);
}

NOINLINE void World::ApplyGravity()
{
    MICROPROFILE_SCOPEI("Physics", "ApplyGravity", -1);

    for (size_t bodyIndex = 0; bodyIndex < bodies.size(); bodyIndex++)
    {
        RigidBody* body = &bodies[bodyIndex];

        if (body->invMass > 0.0f)
        {
            body->acceleration.y += gravity;
        }
    }
}

NOINLINE void World::IntegrateVelocity(float dt)
{
    MICROPROFILE_SCOPEI("Physics", "IntegrateVelocity", -1);

    for (RigidBody& body: bodies)
    {
        body.velocity += body.acceleration * dt;
        body.acceleration = Vector2f(0.0f, 0.0f);

        body.angularVelocity += body.angularAcceleration * dt;
        body.angularAcceleration = 0.0f;
    }
}

NOINLINE void World::IntegratePosition(float dt)
{
    MICROPROFILE_SCOPEI("Physics", "IntegratePosition", -1);

    for (RigidBody& body: bodies)
    {
        body.coords.pos += body.displacingVelocity + body.velocity * dt;
        body.coords.Rotate(-(body.displacingAngularVelocity + body.angularVelocity * dt));

        body.displacingVelocity = Vector2f(0.0f, 0.0f);
        body.displacingAngularVelocity = 0.0f;

        body.UpdateGeom();
    }
}

NOINLINE void World::RefreshContactJoints()
{
    MICROPROFILE_SCOPEI("Physics", "RefreshContactJoints", -1);

    for (size_t jointIndex = 0; jointIndex < solver.contactJoints.size(); jointIndex++)
    {
        solver.contactJoints[jointIndex].contactPointIndex = -1;
    }

    for (size_t manifoldIndex = 0; manifoldIndex < collider.manifolds.size(); ++manifoldIndex)
    {
        Manifold& man = collider.manifolds[manifoldIndex];

        for (int collisionIndex = 0; collisionIndex < man.pointCount; collisionIndex++)
        {
            int contactPointIndex = man.pointIndex + collisionIndex;
            ContactPoint& col = collider.contactPoints[contactPointIndex];

            if (col.solverIndex < 0)
            {
                col.solverIndex = solver.contactJoints.size();

                solver.contactJoints.push_back(ContactJoint(man.body1->index, man.body2->index, contactPointIndex));
            }
            else
            {
                ContactJoint& joint = solver.contactJoints[col.solverIndex];

                assert(joint.body1Index == man.body1->index);
                assert(joint.body2Index == man.body2->index);

                joint.contactPointIndex = contactPointIndex;
            }
        }
    }

    for (size_t jointIndex = 0; jointIndex < solver.contactJoints.size();)
    {
        ContactJoint& joint = solver.contactJoints[jointIndex];

        if (joint.contactPointIndex < 0)
        {
            joint = solver.contactJoints.back();
            solver.contactJoints.pop_back();
        }
        else
        {
            collider.contactPoints[joint.contactPointIndex].solverIndex = jointIndex;
            jointIndex++;
        }
    }
}