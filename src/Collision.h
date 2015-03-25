#pragma once
#include "Vector2.h"
#include "Geom.h"

struct Collision
{
  Collision()
  {
  }
  Collision(Vector2f point1, const Vector2f &point2, const Vector2f normal, RigidBody *body1, RigidBody *body2)
  {
    this->delta1 = point1 - body1->coords.pos;
    this->delta2 = point2 - body2->coords.pos;
    this->normal = normal;
    isMerged = 0;
    isNewlyCreated = 1;
    userInfo = 0;
  }

  bool Equals(Collision *other, float tolerance)
  {
    if (((other->delta1 - delta1).SquareLen() > tolerance * tolerance) &&
      ((other->delta2 - delta2).SquareLen() > tolerance * tolerance))
    {
      return 0;
    }
    return 1;
  }

  void Refresh(Collision *col)
  {
    isMerged = 1;
    isNewlyCreated = 0;
    normal = col->normal;
    delta1 = col->delta1;
    delta2 = col->delta2;
  }

  Vector2f delta1, delta2;
  Vector2f normal;
  bool isMerged;
  bool isNewlyCreated;
  void *userInfo;
};