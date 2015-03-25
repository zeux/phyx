#pragma once
template<typename T>
class Coords2
{
public:
  Vector2<T> xVector, yVector;
  Vector2<T> pos;
  Coords2(){}
  Coords2(const Vector2<T> &_pos, const T angle)
  {
    float pi = 3.141592f;
    xVector = Vector2<T>(cos(angle), sin(angle));
    yVector = Vector2<T>(cos(angle + T(pi) / T(2.0)), sin(angle + T(pi) / T(2.0)));

    pos = _pos;
  }

  const Vector2<T> GetPointRelativePos(const Vector2<T> &globalPoint) const
  {
    Vector2<T> delta = globalPoint - pos;
    return Vector2<T>(delta * xVector, 
            delta * yVector);
  }

  const Vector2<T> GetAxisRelativeOrientation(const Vector2<T> &globalAxis) const
  {
    Vector2<T> delta = globalAxis;
    return Vector2<T>(delta * xVector, 
            delta * yVector);
  }

  const Vector2<T> GetPointGlobalPos  (const Vector2<T> &relativePoint) const
  {
    return pos + xVector * relativePoint.x + yVector * relativePoint.y;
  }

  const Vector2<T> GetAxisGlobalOrientation  (const Vector2<T> &relativeAxis) const
  {
    return xVector * relativeAxis.x + yVector * relativeAxis.y;
  }

  const Coords2<T> GetGlobalCoords(const Coords2<T> &localCoords)
  {
    Coords2<T> res;
    res.pos = GetPointGlobalPos(localCoords.pos);
    res.xVector = GetAxisGlobalOrientation(localCoords.xVector);
    res.yVector = GetAxisGlobalOrientation(localCoords.yVector);
    return res;
  }

  const Coords2<T> GetLocalCoords(const Coords2<T> &globalCoords)
  {
    Coords2<T> res;
    res.pos = GetPointRelativePos(globalCoords.pos);
    res.xVector = GetAxisRelativeOrientation(globalCoords.xVector);
    res.yVector = GetAxisRelativeOrientation(globalCoords.yVector);
    return res;
  }

  void Identity()
  {
    xVector = Vector2<T>(1.0f, 0.0f);
    yVector = Vector2<T>(0.0f, 1.0f);
    pos = Vector2<T>::zeroVector();
  }

  void SetRotation(const T &angle)
  {
    xVector = Vector2<T>(cos(angle), sin(angle));
    yVector = Vector2<T>(cos(angle + pi / 2.0), sin(angle + pi / 2.0));
  }

  void Rotate(const T &angle)
  {
    this->xVector.Rotate(angle);
    this->yVector.Rotate(angle);
  }

  static const Coords2<T> defCoords()
  {
    Coords2<T> coords;
    coords.pos = Vector2<T>::zeroVector();
    coords.xVector = Vector2<T>::xAxis();
    coords.yVector = Vector2<T>::yAxis();
    return coords;
  }
};

typedef Coords2<float>	Coords2f;
typedef Coords2<double>	Coords2d;

typedef Coords2f Coords2f;
typedef Coords2d Coords2d;
