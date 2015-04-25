#pragma once

#include "RigidBody.h"
#include "Collider.h"
#include "Solver.h"

struct Configuration;

struct World
{
    enum SolveMode
    {
        Solve_Scalar,
        Solve_SSE2,
        Solve_AVX2,
    };

    World();

    RigidBody* AddBody(Coords2f coords, Vector2f size);

    void Update(WorkQueue& queue, float dt, const Configuration& configuration);

    NOINLINE void IntegrateVelocity(WorkQueue& queue, float dt);
    NOINLINE void IntegratePosition(WorkQueue& queue, float dt);
    NOINLINE void RefreshContactJoints();

    float collisionTime;
    float mergeTime;
    float solveTime;

    AlignedArray<RigidBody> bodies;
    Collider collider;
    Solver solver;

    float gravity;
};