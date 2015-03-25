#pragma once

template<typename T>
class AABB2
{
public:
  AABB2()
  {
    Reset();
  }
  AABB2(const Vector2<T> &_boxPoint1, const Vector2<T> &_boxPoint2)
  {
    Set(_boxPoint1, _boxPoint2);
  }

  bool Intersects(const AABB2<T> &aabb) const
  {
    if((boxPoint1.x > aabb.boxPoint2.x) || (aabb.boxPoint1.x > boxPoint2.x)) return 0;
    if((boxPoint1.y > aabb.boxPoint2.y) || (aabb.boxPoint1.y > boxPoint2.y)) return 0;
    return 1;
  }

  template<bool isFiniteCast>
  bool Intersects(Vector2<T> rayStart, Vector2<T> rayDir, float &paramMin, float &paramMax)
  {
    // r.dir is unit direction vector of ray
    Vector2f invDir(1.0f / rayDir.x, 1.0f / rayDir.y);

    // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
    // r.org is origin of ray
    float t1 = (boxPoint1.x - rayStart.x) * invDir.x;
    float t2 = (boxPoint2.x - rayStart.x) * invDir.x;
    float t3 = (boxPoint1.y - rayStart.y) * invDir.y;
    float t4 = (boxPoint2.y - rayStart.y) * invDir.y;

    paramMin = std::max(std::min(t1, t2), std::min(t3, t4));
    paramMax = std::min(std::max(t1, t2), std::max(t3, t4));

    // if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behing us
    if (isFiniteCast && paramMax < 0)
    {
        return false;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (paramMin > paramMax)
    {
        return false;
    }
    return true;
  }
  bool Includes(const Vector2<T> &point) const
  {
    if ((point.x < boxPoint1.x) || (point.x > boxPoint2.x) ||
      (point.y < boxPoint1.y) || (point.y > boxPoint2.y))
    {
      return 0;
    }
    return 1;
  }
  bool Includes(const AABB2<T> &aabb) const
  {
    return Includes(aabb.boxPoint1) && Includes(aabb.boxPoint2);
  }
  void Set(const Vector2<T> &_boxPoint1, const Vector2<T> &_boxPoint2)
  {
    boxPoint1 = _boxPoint1;
    boxPoint2 = _boxPoint2;
  }
  void Reset()
  {
    boxPoint1 = Vector2<T>::zeroVector();
    boxPoint2 = Vector2<T>::zeroVector();
  }
  void Expand(const Vector2<T> &additionalPoint)
  {
    boxPoint1.x = Min(boxPoint1.x, additionalPoint.x);
    boxPoint1.y = Min(boxPoint1.y, additionalPoint.y);

    boxPoint2.x = Max(boxPoint2.x, additionalPoint.x);
    boxPoint2.y = Max(boxPoint2.y, additionalPoint.y);
  }
  void Expand(const AABB2<T> &internalAABB)
  {
    Expand(internalAABB.boxPoint1);
    Expand(internalAABB.boxPoint2);
  }
  void Expand(const T mult)
  {
    Vector2<T> size = boxPoint2 - boxPoint1;
    boxPoint1 -= size * (mult - T(1.0)) * T(0.5);
    boxPoint2 += size * (mult - T(1.0)) * T(0.5);
  }
  const T Square() //perimeter actually, used for AABBTree balancing
  {
    Vector2<T> size = boxPoint2 - boxPoint1;
    return (size.x + size.y) * T(2.0);
  }
  Vector2<T> GetSize()
  {
    return boxPoint2 - boxPoint1;
  }
  Vector2<T> boxPoint1, boxPoint2;
};

typedef AABB2<float>	AABB2f;
typedef AABB2<double>	AABB2d;

