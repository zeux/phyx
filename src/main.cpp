#include <SFML/Graphics.hpp>
#include "PhysSystem.h"
#include <sstream>
#include <iomanip>

//50 125 218
//50 50 50

sf::Vector2f ConvertVector(Vector2f vec)
{
  return sf::Vector2f(vec.x, vec.y);
}

void RenderBox(sf::RenderWindow *window, Coords2f coords, Vector2f size, sf::Color color)
{
  Vector2f localPoints[4];
  localPoints[0] = Vector2f(-size.x, -size.y);
  localPoints[1] = Vector2f(size.x, -size.y);
  localPoints[2] = Vector2f(size.x, size.y);
  localPoints[3] = Vector2f(-size.x, size.y);

  sf::Vertex vertices[4];
  for (int vertexNumber = 0; vertexNumber < 4; vertexNumber++)
  {
    vertices[vertexNumber].color = color;
    vertices[vertexNumber].position = ConvertVector(coords.GetPointGlobalPos(localPoints[vertexNumber]));
  }

  window->draw(vertices, 4, sf::Quads);
}

float random(float min, float max)
{
  return min + (max - min) * (float(rand()) / float(RAND_MAX));
}

const struct { PhysSystem::SolveMode mode; const char* name; } kModes[] =
{
    { PhysSystem::Solve_Baseline, "Baseline" },
    { PhysSystem::Solve_AoS, "AoS" },
    { PhysSystem::Solve_SoA_Scalar, "SoA Scalar" },
    { PhysSystem::Solve_SoA_SSE2, "SoA SSE2" },

  #ifdef __AVX2__
    { PhysSystem::Solve_SoA_AVX2, "SoA AVX2" },
  #endif

    { PhysSystem::Solve_SoAPacked_Scalar, "SoA Packed Scalar" },
    { PhysSystem::Solve_SoAPacked_SSE2, "SoA Packed SSE2" },

  #ifdef __AVX2__
    { PhysSystem::Solve_SoAPacked_AVX2, "SoA Packed AVX2" },
  #endif
};

int main(int argc, char** argv)
{
  int windowWidth = 1024, windowHeight = 768;

  PhysSystem physSystem;
  RigidBody *groundBody = physSystem.AddBody(Coords2f(Vector2f(windowWidth * 0.5f, windowHeight * 0.95f), 0.0f), Vector2f(windowWidth * 10.45f, 10.0f));
  groundBody->invInertia = 0.0f;
  groundBody->invMass = 0.0f;

  int currentMode = sizeof(kModes) / sizeof(kModes[0]) - 1;

  const float gravity = 200.0f;
  const float integrationTime = 2e-2f;

  float physicsTime = 0.0f;

  RigidBody *draggedBody = physSystem.AddBody(
    Coords2f(Vector2f(windowWidth * 0.1f, windowHeight * 0.7f), 0.0f), Vector2f(30.0f, 30.0f));

  for (int bodyIndex = 0; bodyIndex < 1000; bodyIndex++)
  {
    RigidBody *testBody = physSystem.AddBody(
      Coords2f(Vector2f(windowWidth * 0.5f, windowHeight * 0.6f) + Vector2f(random(-250.0f, 250.0f), random(-650.0f, 250.0f)), 0.0f), Vector2f(15.0f, 15.0f));
    //testBody->invInertia = 0;
    testBody->velocity = Vector2f(10.0f, 0.0f);
  }

  if (argc > 1 && strcmp(argv[1], "profile") == 0)
  {
    for (int mode = 0; mode < sizeof(kModes) / sizeof(kModes[0]); ++mode)
    {
      PhysSystem testSystem;

      for (int i = 0; i < physSystem.GetBodiesCount(); ++i)
      {
        RigidBody* body = physSystem.GetBody(i);

        RigidBody* testBody = testSystem.AddBody(body->coords, body->geom.size);
        testBody->velocity = body->velocity;
      }

      double solveTime = 0;

      for (int i = 0; i < 10; ++i)
      {
        testSystem.Update(1.f / 60.f, kModes[mode].mode);

        solveTime += testSystem.solveTime;
      }

      printf("%s: %.2f ms\n", kModes[mode].name, solveTime * 1000.f);
    }

    return 0;
  }

  sf::Font font;
  font.loadFromFile("DroidSansMono.ttf");

  sf::RenderWindow *window = new sf::RenderWindow(sf::VideoMode(windowWidth, windowHeight), "This is awesome!");

  sf::Clock clock;

  float prevUpdateTime = 0.0f;

  bool paused = 0;

  while (window->isOpen())
  {
    sf::Event event;
    while (window->pollEvent(event))
    {
      if (event.type == sf::Event::Closed)
        window->close();
      else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::M)
        currentMode = (currentMode + 1) % (sizeof(kModes) / sizeof(kModes[0]));
    }
    Vector2f mousePos = Vector2f(sf::Mouse::getPosition(*window).x, sf::Mouse::getPosition(*window).y);
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
    {
      paused = 0;
    }
    window->clear(sf::Color(50, 50, 50, 255));
    if (clock.getElapsedTime().asSeconds() > prevUpdateTime + integrationTime)
    {
      sf::Clock physicsClock;
      prevUpdateTime += integrationTime;
      if (!paused)
      {

        for (size_t bodyIndex = 0; bodyIndex < physSystem.GetBodiesCount(); bodyIndex++)
        {
          physSystem.GetBody(bodyIndex)->acceleration = Vector2f(0.0f, 0.0f);
          physSystem.GetBody(bodyIndex)->angularAcceleration = 0.0f;
        }
        for (size_t bodyIndex = 0; bodyIndex < physSystem.GetBodiesCount(); bodyIndex++)
        {
          RigidBody *body = physSystem.GetBody(bodyIndex);
          if (body->invMass > 0.0f)
          {
            physSystem.GetBody(bodyIndex)->acceleration += Vector2f(0.0f, gravity);
          }
        }
        RigidBody *draggedBody = physSystem.GetBody(1);
        Vector2f dstVelocity = (mousePos - draggedBody->coords.pos) * 5e1f;
        draggedBody->acceleration += (dstVelocity - draggedBody->velocity) * 5e0;

        physSystem.Update(integrationTime, kModes[currentMode].mode);
        physicsTime = physicsClock.getElapsedTime().asSeconds();
      }
    }


    for (size_t bodyIndex = 0; bodyIndex < physSystem.GetBodiesCount(); bodyIndex++)
    {
      RigidBody *body = physSystem.GetBody(bodyIndex);
      Coords2f bodyCoords = body->coords;
      Vector2f size = body->geom.size;

      float colorMult = float(bodyIndex) / float(physSystem.GetBodiesCount()) * 0.5f + 0.5f;
      sf::Color color = sf::Color(char(50 * colorMult), char(125 * colorMult), char(218 * colorMult));

      if (bodyIndex == 1) //dragged body
      {
        color = sf::Color(242, 236, 164, 255);
      }

      RenderBox(window, bodyCoords, size, color);
    }


    bool pickingCollision = 0;
    /*if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
    {
      pickingCollision = true;
    }*/
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::C))
    for (Collider::ManifoldMap::iterator man = physSystem.GetCollider()->manifolds.begin(); man != physSystem.GetCollider()->manifolds.end(); man++)
    {
      for (int collisionNumber = 0; collisionNumber < man->second.collisionsCount; collisionNumber++)
      {
        Coords2f coords = Coords2f(Vector2f(0.0f, 0.0f), 3.1415f / 4.0f);

        coords.pos = man->second.body1->coords.pos + man->second.collisions[collisionNumber].delta1;

        float redMult = 1.0f;
        if (man->second.collisions[collisionNumber].isNewlyCreated)
          redMult = 0.5f;

        sf::Color color1(100, char(100 * redMult), char(100 * redMult), 100);
        RenderBox(window, coords, Vector2f(3.0f, 3.0f), color1);

        coords.pos = man->second.body2->coords.pos + man->second.collisions[collisionNumber].delta2;
        sf::Color color2(150, char(150 * redMult), char(150 * redMult), 100);
        //if (pickingCollision && (coords.pos - mousePos).SquareLen() < 5.0f * 5.0f)
        //{
        //  color2 = sf::Color(255, 0, 0, 255);
        //}
        RenderBox(window, coords, Vector2f(3.0f, 3.0f), color2);
      }
    }

    std::stringstream debugTextStream;
    debugTextStream << "Bodies count: " << physSystem.GetBodiesCount() << " contacts count: " << physSystem.GetJointsCount();
    sf::Text text(debugTextStream.str().c_str(), font, 20);
    text.setPosition(sf::Vector2f(10.0f, 30.0f));
    window->draw(text);

    std::stringstream debugTextStream2;
    debugTextStream2 << std::fixed;
    debugTextStream2.precision(2);
    debugTextStream2 << 
      "Mode: " << kModes[currentMode].name << "; " <<
      "Physics time: " << std::setw(5) << physicsTime * 1000.0f << 
      "ms (c: " << std::setw(5) << physSystem.collisionTime * 1000.0f << 
      "ms, m: " << std::setw(5) << physSystem.mergeTime * 1000.0f << 
      "ms, s: " << std::setw(5) << physSystem.solveTime * 1000.0f << "ms)";
    sf::Text text2(debugTextStream2.str().c_str(), font, 20);
    text2.setPosition(sf::Vector2f(10.0f, 50.0f));
    window->draw(text2);


    window->display();
  }
}