#pragma once
#include "Vector2.h"
#include "Coords2.h"
#include "AABB2.h"

struct RigidBody;
struct Geom
{
  Vector2f GetClippingVertex(const Vector2f &axis) const
  {
    Vector2f xdim = coords.xVector * size.x;
    Vector2f ydim = coords.yVector * size.y;

    float xsgn = coords.xVector * axis < 0.0f ? -1.0f : 1.0f;
    float ysgn = coords.yVector * axis < 0.0f ? -1.0f : 1.0f;

    return coords.pos + xsgn * xdim + ysgn * ydim;
  }

  bool GetClippingEdge(const Vector2f &axis, Vector2f &edgepoint1, Vector2f &edgepoint2) const
  {
    edgepoint1 = coords.pos;
    edgepoint2 = coords.pos;
    Vector2f xdim = coords.xVector * size.x;
    Vector2f ydim = coords.yVector * size.y;
    Vector2f offset = Vector2f::zero();
    float xdiff = axis * coords.xVector;
    float ydiff = axis * coords.yVector;

    if (fabsf(xdiff) < fabsf(ydiff))
    {
      if (axis * ydim > 0.0f)
      {
        offset += ydim;
        edgepoint1 += xdim;
        edgepoint2 -= xdim;
      }
      else
      {
        offset -= ydim;
        edgepoint1 -= xdim;
        edgepoint2 += xdim;
      }
    }
    else
    {
      if (axis * xdim > 0.0f)
      {
        offset += xdim;
        edgepoint1 -= ydim;
        edgepoint2 += ydim;
      }
      else
      {
        offset -= xdim;
        edgepoint1 += ydim;
        edgepoint2 -= ydim;
      }
    }
    edgepoint1 += offset;
    edgepoint2 += offset;
    return 1;
  }

  int GetSupportPointSet(const Vector2f &axis, Vector2f *supportPoints)
  {
    if (
      (fabsf(axis * coords.xVector) < 0.1f) ||
      (fabsf(axis * coords.yVector) < 0.1f))
    {
      GetClippingEdge(axis, supportPoints[0], supportPoints[1]);
      return 2;
    }

    supportPoints[0] = GetClippingVertex(axis);
    return 1;
  }

  void GetAxisProjectionRange(const Vector2f &axis, float &min, float &max) const
  {
    float sqrlen = axis.SquareLen();
    float invsqrlen = 1.0f;
    invsqrlen = 1.0f / sqrlen;

    float diff =
      fabsf(coords.xVector * axis) * size.x * invsqrlen +
      fabsf(coords.yVector * axis) * size.y * invsqrlen;
    float base = coords.pos * axis * invsqrlen;
    min = base - diff;
    max = base + diff;
  }

  void RecomputeAABB()
  {
    aabb.Reset();
    Vector2f diff = Vector2f(
      fabsf(coords.xVector.x) * size.x + fabsf(coords.yVector.x) * size.y,
      fabsf(coords.xVector.y) * size.x + fabsf(coords.yVector.y) * size.y);
    aabb.Set(coords.pos - diff, coords.pos + diff);
  }

  Vector2f size;
  Coords2f coords;
  AABB2f aabb;
};