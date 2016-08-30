#pragma once

#include "Vector2.h"
#include "Geom.h"
#include "RigidBody.h"

#include <limits>
#include <cassert>

static const int kMaxContactPoints = 2;

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
        body1Index = -1;
        body2Index = -1;
        pointCount = 0;
        pointIndex = 0;
    }

    Manifold(int body1Index, int body2Index, int pointIndex)
    {
        this->body1Index = body1Index;
        this->body2Index = body2Index;
        this->pointCount = 0;
        this->pointIndex = pointIndex;
    }

    int body1Index;
    int body2Index;

    int pointCount;
    int pointIndex;
};