#pragma once

#include "RigidBody.h"
#include "Collider.h"
#include "Solver.h"

#include <vector>

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

    void Update(WorkQueue& queue, float dt, SolveMode mode, int contactIterationsCount, int penetrationIterationsCount);

    NOINLINE void ApplyGravity();
    NOINLINE void IntegrateVelocity(float dt);
    NOINLINE void IntegratePosition(float dt);
    NOINLINE void RefreshContactJoints();

    float collisionTime;
    float mergeTime;
    float solveTime;
    float iterations;

    std::vector<RigidBody> bodies;
    Collider collider;
    Solver solver;

    float gravity;
};