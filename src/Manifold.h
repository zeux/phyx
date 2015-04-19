#pragma once

#include "Vector2.h"
#include "Geom.h"
#include "Collision.h"

#include <limits>
#include <cassert>

struct Manifold
{
    Manifold()
    {
        body1 = body2 = 0;
        collisionsCount = -1;
    }
    Manifold(RigidBody* body1, RigidBody* body2)
    {
        this->body1 = body1;
        this->body2 = body2;
        collisionsCount = 0;
    }

    RigidBody* body1;
    RigidBody* body2;
    int collisionsCount;
    Collision collisions[4]; //in 2d there's always 2 collisions max and 2 more may occur temporarily before merging
};