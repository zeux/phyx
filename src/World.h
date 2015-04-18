#pragma once

#include "RigidBody.h"
#include "Collider.h"
#include "Solver.h"

#include <vector>

struct World
{
    enum SolveMode
    {
        Solve_Baseline,
        Solve_AoS,
        Solve_SoA_Scalar,
        Solve_SoA_SSE2,
        Solve_SoA_AVX2,
        Solve_SoAPacked_Scalar,
        Solve_SoAPacked_SSE2,
        Solve_SoAPacked_AVX2,
        Solve_SoAPacked_FMA,
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