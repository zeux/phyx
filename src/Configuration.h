#pragma once

struct Configuration
{
    enum SolveMode
    {
        Solve_Scalar,
        Solve_SSE2,
        Solve_AVX2,
    };

    enum IslandMode
    {
    	Island_Single,
    	Island_Multiple,
    	Island_SingleSloppy,
    	Island_MultipleSloppy
    };

    SolveMode solveMode;
    IslandMode islandMode;
    int contactIterationsCount;
    int penetrationIterationsCount;
};