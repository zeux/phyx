#pragma once

#include "RigidBody.h"
#include "Collision.h"
#include <algorithm>
struct Limiter
{
  Limiter()
  {
    accumulatedImpulse = 0.0f;
  }
  Vector2f normalProjector1, normalProjector2;
  float  angularProjector1, angularProjector2;

  Vector2f compMass1_linear;
  Vector2f compMass2_linear;
  float  compMass1_angular;
  float  compMass2_angular;
  float compInvMass;

  inline void Refresh(const Vector2f &n1, const Vector2f &n2, const Vector2f &w1, const Vector2f &w2, RigidBody *body1, RigidBody *body2)
  {
    SetJacobian(n1, n2, n1 ^ w1, n2 ^ w2, body1, body2);
  }
  inline void SetJacobian(
    const Vector2f& normalProjector1, const Vector2f& normalProjector2, const float& angularProjector1, const float& angularProjector2,
    RigidBody *body1, RigidBody *body2)
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
    compMass1 = normalProjector1  * compMass1_linear;
    compMass1 += angularProjector1 * compMass1_angular;

    float compMass2;
    compMass2 = normalProjector2  * compMass2_linear;
    compMass2 += angularProjector2 * compMass2_angular;
    //compInvMass = 1.0f / (compMass);
    float compMass = compMass1 + compMass2;

    //accumulatedImpulse = 0;

    if (fabsf(compMass)  > 0) this->compInvMass = 1.0f / (compMass);  else this->compInvMass = 0.0f;
  }

  void  PreStep(RigidBody *body1, RigidBody *body2)
  {
    ApplyImpulse(body1->velocity, body1->angularVelocity, body2->velocity, body2->angularVelocity, accumulatedImpulse);
  }

  float ComputeDeltaImpulse(RigidBody *body1, RigidBody *body2, float dstVelocity)
  {
    return Limiter::ComputeDeltaImpulse(
      body1->velocity, body2->velocity, body1->angularVelocity, body2->angularVelocity, dstVelocity);
  }
  void ApplyImpulse(RigidBody *body1, RigidBody *body2, float deltaImpulse)
  {
    Limiter::ApplyImpulse(
      body1->velocity,
      body1->angularVelocity,
      body2->velocity,
      body2->angularVelocity,
      deltaImpulse);
  }

  float accumulatedImpulse;
protected:
  inline void ApplyImpulse(
    Vector2f &body1Velocity, float &body1AngularVelocity,
    Vector2f &body2Velocity, float &body2AngularVelocity, const float deltaImpulse)
  {
    body1Velocity += compMass1_linear * deltaImpulse;
    body1AngularVelocity += compMass1_angular * deltaImpulse;
    body2Velocity += compMass2_linear * deltaImpulse;
    body2AngularVelocity += compMass2_angular * deltaImpulse;
  }

  float ComputeDeltaImpulse(
    Vector2f body1Velocity,
    Vector2f body2Velocity,
    float body1AngularVelocity,
    float body2AngularVelocity,
    const float dstVelocity)
  {
    float dV = 0;
    dV -= normalProjector1 * body1Velocity;
    dV -= angularProjector1 * body1AngularVelocity;
    dV -= normalProjector2 * body2Velocity;
    dV -= angularProjector2 * body2AngularVelocity;
    dV += dstVelocity;

    return dV * compInvMass;
  }
};

struct FrictionLimiter : public Limiter
{
  void Refresh(const Vector2f &fdir, const Vector2f &point1, const Vector2f &point2, RigidBody *body1, RigidBody *body2)
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
  inline void Refresh(
    const Vector2f &normal, const Vector2f &point1, const Vector2f &point2,
    RigidBody *body1, RigidBody *body2, const float bounce, const float deltaVelocity, const float maxPenetrationVelocity, float deltaDepth, float errorReduction)
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

  float ComputeDeltaDisplacingImpulse(RigidBody *body1, RigidBody *body2)
  {
    return Limiter::ComputeDeltaImpulse(
      body1->displacingVelocity, body2->displacingVelocity, body1->displacingAngularVelocity, body2->displacingAngularVelocity, 0);
  }
  void ApplyDisplacingImpulse(RigidBody *body1, RigidBody *body2, float deltaImpulse)
  {
    Limiter::ApplyImpulse(
      body1->displacingVelocity,
      body1->displacingAngularVelocity,
      body2->displacingVelocity,
      body2->displacingAngularVelocity,
      deltaImpulse);
  }

  float SolveImpulse(RigidBody *body1, RigidBody *body2)
  {
    float deltaImpulse = Limiter::ComputeDeltaImpulse(body1, body2, dstVelocity);
    if (deltaImpulse + accumulatedImpulse < 0.0f) deltaImpulse = -accumulatedImpulse;
    ApplyImpulse(body1, body2, deltaImpulse);
    accumulatedImpulse += deltaImpulse;
    return deltaImpulse;
  }
  float SolveDisplacingImpulse(RigidBody *body1, RigidBody *body2)
  {
    float deltaDisplacingImpulse = Limiter::ComputeDeltaImpulse(
      body1->displacingVelocity,
      body2->displacingVelocity,
      body1->displacingAngularVelocity,
      body2->displacingAngularVelocity,
      dstDisplacingVelocity);
    if (deltaDisplacingImpulse + accumulatedDisplacingImpulse < 0.0f) deltaDisplacingImpulse = -accumulatedDisplacingImpulse;
    ApplyImpulse(
      body1->displacingVelocity,
      body1->displacingAngularVelocity,
      body2->displacingVelocity,
      body2->displacingAngularVelocity,
      deltaDisplacingImpulse);
    accumulatedDisplacingImpulse += deltaDisplacingImpulse;
    return deltaDisplacingImpulse;
  }
  float dstVelocity;
  float dstDisplacingVelocity;
  float accumulatedDisplacingImpulse;
};

struct ContactJoint
{
  struct Descriptor
  {
    Collision *collision;
    RigidBody *body1;
    RigidBody *body2;
  };
  ContactJoint(const Descriptor &desc)
  {
    this->valid = 1;
    this->collision = desc.collision;
    this->body1 = desc.body1;
    this->body2 = desc.body2;
    collision->userInfo = this;

  }
  void Refresh()
  {
    collision->userInfo = this;

    Vector2f w1 = collision->delta1;
    Vector2f w2 = collision->delta2;


    Vector2f point1 = w1 + body1->coords.pos;
    Vector2f point2 = w2 + body2->coords.pos;
    normalLimiter.Refresh(collision->normal, point1, point2, body1, body2, 0.0f, 1.0f, 0.1f, 1.0f, 0.1f);

    Vector2f tangent;
    tangent = collision->normal.GetPerpendicular();
    frictionLimiter.Refresh(tangent, point1, point2, body1, body2);
  }
  void PreStep()
  {
    normalLimiter.PreStep(body1, body2);
    frictionLimiter.PreStep(body1, body2);
  }
  void SolveImpulse()
  {
    normalLimiter.SolveImpulse(body1, body2);
    //if(!shock)
    /*float deltaImpulse1 = fdir1.ComputeDeltaImpulse<fixed1, fixed2>(body1->velocity, body1->angularVelocity, body2->velocity, body2->angularVelocity, 0);
    float deltaImpulse2 = fdir2.ComputeDeltaImpulse<fixed1, fixed2>(body1->velocity, body1->angularVelocity, body2->velocity, body2->angularVelocity, 0);*/
    float deltaImpulse =
      frictionLimiter.ComputeDeltaImpulse(body1, body2, 0.0f);

    float reactionForce;
    float accumulatedImpulse;

    reactionForce = normalLimiter.accumulatedImpulse;
    accumulatedImpulse = frictionLimiter.accumulatedImpulse;

    float frictionForce = accumulatedImpulse + deltaImpulse;
    float frictionCoefficient = 0.3f;
    if (fabs(frictionForce) > (reactionForce * frictionCoefficient))
    {
      float dir = frictionForce > 0.0f ? 1.0f : -1.0f;
      frictionForce = dir * reactionForce * frictionCoefficient;
      deltaImpulse = frictionForce - accumulatedImpulse;
    }
    //totalError = max(totalError, fabsf(fdir1->deltaImpulse));
    frictionLimiter.accumulatedImpulse += deltaImpulse;
    frictionLimiter.ApplyImpulse(body1, body2, deltaImpulse);
  }
  void SolveDisplacement()
  {
    normalLimiter.SolveDisplacingImpulse(body1, body2);
  }
  RigidBody *body1;
  RigidBody *body2;
  NormalLimiter normalLimiter;
  FrictionLimiter frictionLimiter;
  Collision *collision;
  bool valid;
};