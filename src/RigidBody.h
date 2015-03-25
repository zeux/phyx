#pragma once
#include "Vector2.h"
#include "Coords2.h"
#include "Geom.h"

struct RigidBody
{
  RigidBody(){} 
  RigidBody(Coords2f coords, Vector2f size, float density)
  {
    this->coords = coords;
    displacingVelocity = Vector2f(0.0f, 0.0f);
    displacingAngularVelocity = 0.0f;

    acceleration = Vector2f(0.0f, 0.0f);
    angularAcceleration = 0.0f;

    velocity = Vector2f(0.0f, 0.0f);
    angularVelocity = 0.0f;

    geom.size = size;

    float mass = density * (size.x * size.y);
    float inertia = mass * (size.x * size.x + size.y * size.y);

    invMass = 1.0f / mass;
    invInertia = 1.0f / inertia;

    UpdateGeom();
  }
  void IntegrateVelocity(float dt)
  {
    velocity += acceleration * dt;
    acceleration = Vector2f(0.0f, 0.0f);

    angularVelocity += angularAcceleration * dt;
    angularAcceleration = 0.0f;
  }
  void IntegratePosition(float dt)
  {
    coords.pos += displacingVelocity + velocity * dt;
    coords.Rotate(-displacingAngularVelocity);
    coords.Rotate(-angularVelocity * dt);

    displacingVelocity = Vector2f(0.0f, 0.0f);
    displacingAngularVelocity = 0.0f;

    UpdateGeom();
  }
  Vector2f GetGlobalPointVelocity(Vector2f point)
  {
    Vector2f relpos = point - coords.pos;
    return Vector2f(-relpos.y, relpos.x) * angularVelocity + velocity;
  }
  void UpdateGeom()
  {
    geom.coords = coords;
    geom.RecomputeAABB();
  }

  Geom geom;

  Vector2f velocity, acceleration;
  Vector2f displacingVelocity;
  float angularVelocity, angularAcceleration;
  float displacingAngularVelocity;

  float invMass, invInertia;
  Coords2f coords;
  char padding[1024];
};