#pragma once

#include "RigidBody.h"
#include "Manifold.h"

struct ContactJoint
{
    ContactJoint(int body1Index, int body2Index, int collisionIndex)
    {
        this->contactPointIndex = collisionIndex;
        this->body1Index = body1Index;
        this->body2Index = body2Index;

        normalLimiter_accumulatedImpulse = 0.f;
        frictionLimiter_accumulatedImpulse = 0.f;
    }

    int contactPointIndex;
    int body1Index;
    int body2Index;
    float normalLimiter_accumulatedImpulse;
    float frictionLimiter_accumulatedImpulse;
};