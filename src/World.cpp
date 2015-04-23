#include "World.h"

#include "base/Parallel.h"
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

    IntegrateVelocity(queue, dt);

    collider.UpdateBroadphase(bodies.data(), bodies.size());
    collider.UpdatePairs(queue, bodies.data(), bodies.size());
    collider.UpdateManifolds(queue, bodies.data());
    collider.PackManifolds(bodies.data());

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

    IntegratePosition(queue, dt);
}

NOINLINE void World::IntegrateVelocity(WorkQueue& queue, float dt)
{
    MICROPROFILE_SCOPEI("Physics", "IntegrateVelocity", -1);

    parallelFor(queue, bodies.data(), bodies.size(), 32, [this, dt](RigidBody& body, int) {
        if (body.invMass > 0.0f)
        {
            body.acceleration.y += gravity;
        }

        body.velocity += body.acceleration * dt;
        body.acceleration = Vector2f(0.0f, 0.0f);

        body.angularVelocity += body.angularAcceleration * dt;
        body.angularAcceleration = 0.0f;
    });
}

NOINLINE void World::IntegratePosition(WorkQueue& queue, float dt)
{
    MICROPROFILE_SCOPEI("Physics", "IntegratePosition", -1);

    parallelFor(queue, bodies.data(), bodies.size(), 32, [dt](RigidBody& body, int) {
        body.coords.pos += body.displacingVelocity + body.velocity * dt;
        body.coords.Rotate(-(body.displacingAngularVelocity + body.angularVelocity * dt));

        body.displacingVelocity = Vector2f(0.0f, 0.0f);
        body.displacingAngularVelocity = 0.0f;

        body.UpdateGeom();
    });
}

NOINLINE void World::RefreshContactJoints()
{
    MICROPROFILE_SCOPEI("Physics", "RefreshContactJoints", -1);

    int matched = 0;
    int created = 0;
    int deleted = 0;

    {
        MICROPROFILE_SCOPEI("Physics", "Reset", -1);

        for (size_t jointIndex = 0; jointIndex < solver.contactJoints.size(); jointIndex++)
        {
            solver.contactJoints[jointIndex].contactPointIndex = -1;
        }
    }

    {
        MICROPROFILE_SCOPEI("Physics", "Match", -1);

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

                    solver.contactJoints.push_back(ContactJoint(man.body1Index, man.body2Index, contactPointIndex));

                    created++;
                }
                else
                {
                    ContactJoint& joint = solver.contactJoints[col.solverIndex];

                    assert(joint.body1Index == man.body1Index);
                    assert(joint.body2Index == man.body2Index);

                    joint.contactPointIndex = contactPointIndex;

                    matched++;
                }
            }
        }
    }

    {
        MICROPROFILE_SCOPEI("Physics", "Cleanup", -1);

        for (size_t jointIndex = 0; jointIndex < solver.contactJoints.size();)
        {
            ContactJoint& joint = solver.contactJoints[jointIndex];

            if (joint.contactPointIndex < 0)
            {
                joint = solver.contactJoints.back();
                solver.contactJoints.pop_back();

                deleted++;
            }
            else
            {
                collider.contactPoints[joint.contactPointIndex].solverIndex = jointIndex;
                jointIndex++;
            }
        }
    }

    MICROPROFILE_META_CPU("Matched", matched);
    MICROPROFILE_META_CPU("Created", created);
    MICROPROFILE_META_CPU("Deleted", deleted);
}