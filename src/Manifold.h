#pragma once
#include "Vector2.h"
#include "Geom.h"
#include "Collision.h"
#include <limits>
struct Manifold
{
  Manifold()
  {
    body1 = body2 = 0;
    isMerged = 0;
    collisionsCount = -1;
  }
  Manifold(RigidBody *body1, RigidBody *body2)
  {
    this->body1 = body1;
    this->body2 = body2;
    isMerged = 1;
    collisionsCount = 0;
  }
  void MergeCollision(Collision *newbie)
  {
    Collision *closest = 0;
    float bestdepth = std::numeric_limits<float>::max();

    for (int collisionIndex = 0; collisionIndex < collisionsCount; collisionIndex++)
    {
      Collision *col = &collisions[collisionIndex];
      if (newbie->Equals(col, 2.0f))
      {
        float depth = (newbie->delta1 - col->delta1).SquareLen() + (newbie->delta2 - col->delta2).SquareLen();
        if (depth < bestdepth)
        {
          bestdepth = depth;
          closest = col;
        }
      }
    }
    if (closest)
    {
      closest->Refresh(newbie);
    }
    else
    {
      assert(collisionsCount < 8);
      newbie->isMerged = 1;
      newbie->isNewlyCreated = 1;
      collisions[collisionsCount++] = *newbie;
    }
  }

  void Update()
  {
    for (int collisionIndex = 0; collisionIndex < collisionsCount; collisionIndex++)
    {
      collisions[collisionIndex].isMerged = 0;
      collisions[collisionIndex].isNewlyCreated = 0;
    }
    Vector2f separatingAxis;
    if (ComputeSeparatingAxis(separatingAxis))
    {
      GenerateContacts(separatingAxis);
    }


    for (int collisionIndex = 0; collisionIndex < collisionsCount;)
    {
      if (!collisions[collisionIndex].isMerged)
      {
        collisions[collisionIndex] = collisions[collisionsCount - 1];
        collisionsCount--;
      }
      else
      {
        collisionIndex++;
      }
    }
  }
  RigidBody *body1;
  RigidBody *body2;
  bool isMerged;
  Collision collisions[4]; //in 2d there's always 2 collisions max and 2 more may occur temporarily before merging
  int collisionsCount;

private:
  bool ComputeSeparatingAxis(Vector2f &separatingAxis)
  {

    Vector2f axis[4];
    axis[0] = body1->coords.xVector;
    axis[1] = body1->coords.yVector;
    axis[2] = body2->coords.xVector;
    axis[3] = body2->coords.yVector;

    bool found = 0;
    float bestquaddepth = 1e5f;
    Vector2f bestaxis;

    float min0;
    float max0;
    for (int i = 0; i < 4; i++)
    {
      float min1, max1;
      float min2, max2;
      body1->geom.GetAxisProjectionRange(axis[i], min1, max1);
      body2->geom.GetAxisProjectionRange(axis[i], min2, max2);
      if ((min1 > max2) || (min2 > max1))
      {
        return 0;
      }
      min0 = std::max(min1, min2);
      max0 = std::min(max1, max2);
      if (min0 > max0) return 0;

      float delta = (std::min(max2 - min1, max1 - min2) * axis[i]).SquareLen();
      if (bestquaddepth > delta)
      {
        bestquaddepth = delta;
        bestaxis = axis[i];
        found = 1;
      }
    }
    separatingAxis = bestaxis;
    return found;
  }
  void GenerateContacts(Vector2f separatingAxis)
  {
    if (separatingAxis * (body1->coords.pos - body2->coords.pos) < 0.0f)
      separatingAxis.Invert();

    const int maxSupportPoints = 2;
    Vector2f supportPoints1[maxSupportPoints];
    Vector2f supportPoints2[maxSupportPoints];

    float linearTolerance = 2.0f;
    float angularTolerance = 0.05f;
    int supportPointsCount1 = body1->geom.GetSupportPointSet(-separatingAxis, supportPoints1);
    int supportPointsCount2 = body2->geom.GetSupportPointSet(separatingAxis, supportPoints2);

    if ((supportPointsCount1 == 2) && (((supportPoints1[0] - supportPoints1[1])).SquareLen() < linearTolerance * linearTolerance))
    {
      supportPoints1[0] = (supportPoints1[0] + supportPoints1[1]) * 0.5f;
      supportPointsCount1 = 1;
    }
    if ((supportPointsCount2 == 2) && (((supportPoints2[0] - supportPoints2[1])).SquareLen() < linearTolerance * linearTolerance))
    {
      supportPoints2[0] = (supportPoints2[0] + supportPoints2[1]) * 0.5f;
      supportPointsCount2 = 1;
    }


    if ((supportPointsCount1 == 1) && (supportPointsCount2 == 1))
    {
      Vector2f delta = supportPoints2[0] - supportPoints1[0];
      //float eps = (delta ^ separatingAxis).SquareLen();
      if (delta * separatingAxis > 0.0f)
      {
        Collision newbie(supportPoints1[0], supportPoints2[0], separatingAxis, body1, body2);
        MergeCollision(&newbie);
      }
    }
    else
      if ((supportPointsCount1 == 1) && (supportPointsCount2 == 2))
      {
        Vector2f n = (supportPoints2[1] - supportPoints2[0]).GetPerpendicular();
        Vector2f point;
        ProjectPointToLine(supportPoints1[0], supportPoints2[0], n, separatingAxis, point);

        if ((((point - supportPoints2[0]) * (supportPoints2[1] - supportPoints2[0])) > 0.0f) &&
          (((point - supportPoints2[1]) * (supportPoints2[0] - supportPoints2[1])) > 0.0f))
        {
          Collision newbie(supportPoints1[0], point, separatingAxis, body1, body2);
          MergeCollision(&newbie);
        }
      }
      else
        if ((supportPointsCount1 == 2) && (supportPointsCount2 == 1))
        {
          Vector2f n = (supportPoints1[1] - supportPoints1[0]).GetPerpendicular();
          Vector2f point;
          ProjectPointToLine(supportPoints2[0], supportPoints1[0], n, separatingAxis, point);

          if ((((point - supportPoints1[0]) * (supportPoints1[1] - supportPoints1[0])) > 0.0f) &&
            (((point - supportPoints1[1]) * (supportPoints1[0] - supportPoints1[1])) > 0.0f))
          {
            Collision newbie(point, supportPoints2[0], separatingAxis, body1, body2);
            MergeCollision(&newbie);
          }
        }
        else
          if ((supportPointsCount2 == 2) && (supportPointsCount1 == 2))
          {
            struct TempColInfo
            {
              Vector2f point1, point2;
            };
            TempColInfo tempCol[4];
            int tempCols = 0;
            for (int i = 0; i < 2; i++)
            {
              Vector2f n = (supportPoints2[1] - supportPoints2[0]).GetPerpendicular();
              if ((supportPoints1[i] - supportPoints2[0]) * n > 0.0)
              {
                Vector2f point;
                ProjectPointToLine(supportPoints1[i], supportPoints2[0], n, separatingAxis, point);


                if ((((point - supportPoints2[0]) * (supportPoints2[1] - supportPoints2[0])) >= 0.0f) &&
                  (((point - supportPoints2[1]) * (supportPoints2[0] - supportPoints2[1])) > 0.0f))
                {
                  tempCol[tempCols].point1 = supportPoints1[i];
                  tempCol[tempCols].point2 = point;
                  tempCols++;
                  //							TryToAdd(new(collisionManager->GetNewNode()) RigidRigid::Collision(supportPoint1[i], point, separatingAxis, proxy1, proxy2));
                }
              }
            }
            for (int i = 0; i < 2; i++)
            {
              Vector2f n = (supportPoints1[1] - supportPoints1[0]).GetPerpendicular();
              if ((supportPoints2[i] - supportPoints1[0]) * n > 0.0)
              {
                Vector2f point;
                ProjectPointToLine(supportPoints2[i], supportPoints1[0], n, separatingAxis, point);

                if ((((point - supportPoints1[0]) * (supportPoints1[1] - supportPoints1[0])) >= 0.0f) &&
                  (((point - supportPoints1[1]) * (supportPoints1[0] - supportPoints1[1])) > 0.0f))
                {
                  tempCol[tempCols].point1 = point;
                  tempCol[tempCols].point2 = supportPoints2[i];
                  tempCols++;
                }
              }
            }

            if (tempCols == 1) //buggy but must work
            {
              Collision newbie(tempCol[0].point1, tempCol[0].point2, separatingAxis, body1, body2);
              MergeCollision(&newbie);
            }
            if (tempCols >= 2) //means only equality, but clamp to two points
            {
              Collision newbie1(tempCol[0].point1, tempCol[0].point2, separatingAxis, body1, body2);
              MergeCollision(&newbie1);
              Collision newbie2(tempCol[1].point1, tempCol[1].point2, separatingAxis, body1, body2);
              MergeCollision(&newbie2);
              //}
            }
          }
  }
};