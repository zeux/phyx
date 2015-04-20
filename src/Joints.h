#pragma once

#include "RigidBody.h"
#include "Manifold.h"
#include <algorithm>
struct Limiter
{
    Limiter()
    {
        accumulatedImpulse = 0.0f;
    }

    Vector2f normalProjector1, normalProjector2;
    float angularProjector1, angularProjector2;

    Vector2f compMass1_linear;
    Vector2f compMass2_linear;
    float compMass1_angular;
    float compMass2_angular;
    float compInvMass;
    float accumulatedImpulse;

    inline void Refresh(const Vector2f& n1, const Vector2f& n2, const Vector2f& w1, const Vector2f& w2, RigidBody* body1, RigidBody* body2)
    {
        SetJacobian(n1, n2, n1 ^ w1, n2 ^ w2, body1, body2);
    }
    inline void SetJacobian(
        const Vector2f& normalProjector1, const Vector2f& normalProjector2, const float& angularProjector1, const float& angularProjector2,
        RigidBody* body1, RigidBody* body2)
    {
        this->normalProjector1 = normalProjector1;
        this->normalProjector2 = normalProjector2;
        this->angularProjector1 = angularProjector1;
        this->angularProjector2 = angularProjector2;

        this->compMass1_linear = normalProjector1 * body1->invMass;
        this->compMass2_linear = normalProjector2 * body2->invMass;
        this->compMass1_angular = angularProjector1 * body1->invInertia;
        this->compMass2_angular = angularProjector2 * body2->invInertia;

        float compMass1;
        compMass1 = normalProjector1 * compMass1_linear;
        compMass1 += angularProjector1 * compMass1_angular;

        float compMass2;
        compMass2 = normalProjector2 * compMass2_linear;
        compMass2 += angularProjector2 * compMass2_angular;

        float compMass = compMass1 + compMass2;

        //accumulatedImpulse = 0;

        if (fabsf(compMass) > 0)
            this->compInvMass = 1.0f / (compMass);
        else
            this->compInvMass = 0.0f;
    }
};

struct FrictionLimiter : public Limiter
{
    void Refresh(const Vector2f& fdir, const Vector2f& point1, const Vector2f& point2, RigidBody* body1, RigidBody* body2)
    {
        Vector2f w1 = point1 - body1->coords.pos;
        Vector2f w2 = point1 - body2->coords.pos;

        Limiter::Refresh(fdir, -fdir, w1, w2, body1, body2);
    }
};

struct NormalLimiter : public Limiter
{
    NormalLimiter()
    {
        accumulatedDisplacingImpulse = 0.0f;
    }

    void Refresh(
        const Vector2f& normal, const Vector2f& point1, const Vector2f& point2,
        RigidBody* body1, RigidBody* body2, const float bounce, const float deltaVelocity, const float maxPenetrationVelocity, float deltaDepth, float errorReduction)
    {
        Vector2f w1 = point1 - body1->coords.pos;
        Vector2f w2 = point1 - body2->coords.pos;
        Limiter::Refresh(normal, -normal, w1, w2, body1, body2);

        dstDisplacingVelocity = 0;

        Vector2f v = body1->GetGlobalPointVelocity(point1);
        v -= body2->GetGlobalPointVelocity(point1);

        float dv = -bounce * (v * normal);

        dstVelocity = std::max(dv - deltaVelocity, 0.0f);

        float depth = (point2 - point1) * normal;
        if (depth < deltaDepth)
        {
            dstVelocity -= maxPenetrationVelocity;
        }

        dstDisplacingVelocity = errorReduction * std::max(0.0f, (depth - 2.0f * deltaDepth));

        accumulatedDisplacingImpulse = 0;
    }

    float dstVelocity;
    float dstDisplacingVelocity;
    float accumulatedDisplacingImpulse;
};

struct ContactJoint
{
    ContactJoint(RigidBody* body1, RigidBody* body2, ContactPoint* collision, int solverIndex)
    {
        this->collision = collision;
        this->body1 = body1;
        this->body2 = body2;
        this->body1Index = body1->index;
        this->body2Index = body2->index;
        collision->solverIndex = solverIndex;
    }

    void Refresh()
    {
        Vector2f w1 = collision->delta1;
        Vector2f w2 = collision->delta2;

        Vector2f point1 = w1 + body1->coords.pos;
        Vector2f point2 = w2 + body2->coords.pos;
        normalLimiter.Refresh(collision->normal, point1, point2, body1, body2, 0.0f, 1.0f, 0.1f, 1.0f, 0.1f);

        Vector2f tangent;
        tangent = collision->normal.GetPerpendicular();
        frictionLimiter.Refresh(tangent, point1, point2, body1, body2);
    }

    ContactPoint* collision;
    RigidBody* body1;
    RigidBody* body2;
    unsigned int body1Index;
    unsigned int body2Index;
    NormalLimiter normalLimiter;
    FrictionLimiter frictionLimiter;
};