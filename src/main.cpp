#include <SFML/Graphics.hpp>
#include "PhysSystem.h"
#include <sstream>
#include <iomanip>
#include <string.h>

sf::Vector2f ConvertVector(Vector2f vec)
{
    return sf::Vector2f(vec.x, vec.y);
}

void RenderBox(std::vector<sf::Vertex> &vertices, Coords2f coords, Vector2f size, sf::Color color)
{
    Vector2f axisX = coords.xVector * size.x;
    Vector2f axisY = coords.yVector * size.y;

    sf::Vertex v;

    v.color = color;

    v.position = ConvertVector(coords.pos - axisX - axisY);
    vertices.push_back(v);

    v.position = ConvertVector(coords.pos + axisX - axisY);
    vertices.push_back(v);

    v.position = ConvertVector(coords.pos + axisX + axisY);
    vertices.push_back(v);

    v.position = ConvertVector(coords.pos - axisX + axisY);
    vertices.push_back(v);
}

float random(float min, float max)
{
    return min + (max - min) * (float(rand()) / float(RAND_MAX));
}

const struct { PhysSystem::SolveMode mode; const char *name; } kModes[] =
{
     {PhysSystem::Solve_Baseline, "Baseline"},
     {PhysSystem::Solve_AoS, "AoS"},
     {PhysSystem::Solve_SoA_Scalar, "SoA Scalar"},
     {PhysSystem::Solve_SoA_SSE2, "SoA SSE2"},

#ifdef __AVX2__
     {PhysSystem::Solve_SoA_AVX2, "SoA AVX2"},
#endif

     {PhysSystem::Solve_SoAPacked_Scalar, "SoA Packed Scalar"},
     {PhysSystem::Solve_SoAPacked_SSE2, "SoA Packed SSE2"},

#ifdef __AVX2__
     {PhysSystem::Solve_SoAPacked_AVX2, "SoA Packed AVX2"},
#endif

#if defined(__AVX2__) && defined(__FMA__)
     {PhysSystem::Solve_SoAPacked_FMA, "SoA Packed FMA"},
#endif
};

int main(int argc, char **argv)
{
    int windowWidth = 1280, windowHeight = 1024;

    std::unique_ptr<WorkQueue> queue(new WorkQueue(WorkQueue::getIdealWorkerCount()));

    PhysSystem physSystem;
    RigidBody *groundBody = physSystem.AddBody(Coords2f(Vector2f(windowWidth * 0.5f, windowHeight * 0.95f), 0.0f), Vector2f(windowWidth * 10.45f, 10.0f));
    groundBody->invInertia = 0.0f;
    groundBody->invMass = 0.0f;

    int currentMode = sizeof(kModes) / sizeof(kModes[0]) - 1;

    const float gravity = 200.0f;
    const float integrationTime = 1 / 60.f;
    const int contactIterationsCount = 15;
    const int penetrationIterationsCount = 15;

    float physicsTime = 0.0f;

    RigidBody *draggedBody = physSystem.AddBody(
        Coords2f(Vector2f(windowWidth * 0.1f, windowHeight * 0.7f), 0.0f), Vector2f(30.0f, 30.0f));

    float bodyRadius = 2.f;
    int bodyCount = 10000;

    for (int bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++)
    {
        RigidBody *testBody = physSystem.AddBody(
            Coords2f(Vector2f(windowWidth * 0.5f, windowHeight * 0.6f) + Vector2f(random(-250.0f, 250.0f), random(-650.0f, 250.0f)), 0.0f), Vector2f(bodyRadius * 2.f, bodyRadius * 2.f));
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
                RigidBody *body = physSystem.GetBody(i);

                RigidBody *testBody = testSystem.AddBody(body->coords, body->geom.size);
                testBody->velocity = body->velocity;
            }

            double collisionTime = 0;
            double mergeTime = 0;
            double solveTime = 0;
            float iterations = 0;

            for (int i = 0; i < 10; ++i)
            {
                testSystem.Update(*queue, 1.f / 60.f, kModes[mode].mode, contactIterationsCount, penetrationIterationsCount);

                collisionTime += testSystem.collisionTime;
                mergeTime += testSystem.mergeTime;
                solveTime += testSystem.solveTime;
                iterations += testSystem.iterations;
            }

            printf("%s: collision %.2f ms, merge %.2f ms, solve %.2f ms, %.2f iterations\n", kModes[mode].name, collisionTime * 1000, mergeTime * 1000, solveTime * 1000, iterations);
        }

        return 0;
    }

    sf::Font font;
    font.loadFromFile("DroidSansMono.ttf");

    sf::RenderWindow *window = new sf::RenderWindow(sf::VideoMode(windowWidth, windowHeight), "This is awesome!");

    sf::Clock clock;

    float prevUpdateTime = 0.0f;

    bool paused = 0;

    std::vector<sf::Vertex> vertices;

    while (window->isOpen())
    {
        sf::Event event;
        while (window->pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window->close();
            else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::M)
                currentMode = (currentMode + 1) % (sizeof(kModes) / sizeof(kModes[0]));
            else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::P)
                paused = !paused;
            else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::C)
            {
                size_t workers = queue->getWorkerCount();
                workers *= 2;
                if (workers > WorkQueue::getIdealWorkerCount())
                    workers = 1;
                queue.reset(new WorkQueue(workers));
            }
        }
        Vector2f mousePos = Vector2f(sf::Mouse::getPosition(*window).x, sf::Mouse::getPosition(*window).y);
        window->clear(sf::Color(50, 50, 50, 255));
        vertices.clear();
        if (clock.getElapsedTime().asSeconds() > prevUpdateTime + integrationTime)
        {
            sf::Clock physicsClock;
            prevUpdateTime += integrationTime;
            float time = 0;
            if (!paused || sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
                time = integrationTime;
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

                physSystem.Update(*queue, time, kModes[currentMode].mode, contactIterationsCount, penetrationIterationsCount);
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

            RenderBox(vertices, bodyCoords, size, color);
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::V))
            for (size_t manifoldIndex = 0; manifoldIndex < physSystem.GetCollider()->manifolds.size(); manifoldIndex++)
            {
                Manifold &man = physSystem.GetCollider()->manifolds[manifoldIndex];

                for (int collisionNumber = 0; collisionNumber < man.collisionsCount; collisionNumber++)
                {
                    Coords2f coords = Coords2f(Vector2f(0.0f, 0.0f), 3.1415f / 4.0f);

                    coords.pos = man.body1->coords.pos + man.collisions[collisionNumber].delta1;

                    float redMult = 1.0f;
                    if (man.collisions[collisionNumber].isNewlyCreated)
                        redMult = 0.5f;

                    sf::Color color1(100, char(100 * redMult), char(100 * redMult), 100);
                    RenderBox(vertices, coords, Vector2f(3.0f, 3.0f), color1);

                    coords.pos = man.body2->coords.pos + man.collisions[collisionNumber].delta2;
                    sf::Color color2(150, char(150 * redMult), char(150 * redMult), 100);

                    RenderBox(vertices, coords, Vector2f(3.0f, 3.0f), color2);
                }
            }

        if (vertices.size() > 0)
            window->draw(&vertices[0], vertices.size(), sf::Quads);

        std::stringstream debugTextStream;
        debugTextStream.precision(2);
        debugTextStream
            << "Bodies: " << physSystem.GetBodiesCount()
            << " Contacts: " << physSystem.GetJointsCount()
            << " Iterations: " << physSystem.iterations;
        sf::Text text(debugTextStream.str().c_str(), font, 20);
        text.setPosition(sf::Vector2f(10.0f, 30.0f));
        window->draw(text);

        std::stringstream debugTextStream2;
        debugTextStream2 << std::fixed;
        debugTextStream2.precision(2);
        debugTextStream2
            << queue->getWorkerCount() << " cores; "
            << "Mode: " << kModes[currentMode].name << "; "
            << "Physics time: " << std::setw(5) << physicsTime * 1000.0f << "ms (c: " << std::setw(5) << physSystem.collisionTime * 1000.0f << "ms, m: " << std::setw(5) << physSystem.mergeTime * 1000.0f << "ms, s: " << std::setw(5) << physSystem.solveTime * 1000.0f << "ms)";
        sf::Text text2(debugTextStream2.str().c_str(), font, 20);
        text2.setPosition(sf::Vector2f(10.0f, 50.0f));
        window->draw(text2);

        window->display();
    }
}