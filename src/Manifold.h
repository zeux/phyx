#pragma once

#include "Vector2.h"
#include "Geom.h"
#include "RigidBody.h"

#include <limits>
#include <cassert>

struct ContactPoint
{
    ContactPoint()
    {
    }

    ContactPoint(Vector2f point1, const Vector2f& point2, const Vector2f normal, RigidBody* body1, RigidBody* body2)
    {
        this->delta1 = point1 - body1->coords.pos;
        this->delta2 = point2 - body2->coords.pos;
        this->normal = normal;
        isMerged = 0;
        isNewlyCreated = 1;
        solverIndex = -1;
    }

    bool Equals(const ContactPoint& other, float tolerance) const
    {
        if (((other.delta1 - delta1).SquareLen() > tolerance * tolerance) &&
            ((other.delta2 - delta2).SquareLen() > tolerance * tolerance))
        {
            return 0;
        }
        return 1;
    }

    Vector2f delta1, delta2;
    Vector2f normal;
    bool isMerged;
    bool isNewlyCreated;
    int solverIndex;
};

struct Manifold
{
    Manifold()
    {
        body1 = body2 = 0;
        pointCount = -1;
    }

    Manifold(RigidBody* body1, RigidBody* body2)
    {
        this->body1 = body1;
        this->body2 = body2;
        pointCount = 0;
    }

    RigidBody* body1;
    RigidBody* body2;

    int pointCount;
    ContactPoint points[4]; //in 2d there's always 2 collisions max and 2 more may occur temporarily before merging
};