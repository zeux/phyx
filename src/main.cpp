#include <GLFW/glfw3.h>

#include "PhysSystem.h"

struct Vertex
{
    Vector2f position;
    unsigned char r, g, b, a;

};

void RenderBox(std::vector<Vertex>& vertices, Coords2f coords, Vector2f size, int r, int g, int b)
{
    Vector2f axisX = coords.xVector * size.x;
    Vector2f axisY = coords.yVector * size.y;

    Vertex v;

    v.r = r;
    v.g = g;
    v.b = b;
    v.a = 255;

    v.position = coords.pos - axisX - axisY;
    vertices.push_back(v);

    v.position = coords.pos + axisX - axisY;
    vertices.push_back(v);

    v.position = coords.pos + axisX + axisY;
    vertices.push_back(v);

    v.position = coords.pos - axisX + axisY;
    vertices.push_back(v);
}

float random(float min, float max)
{
    return min + (max - min) * (float(rand()) / float(RAND_MAX));
}

const struct
{
    PhysSystem::SolveMode mode;
    const char* name;
} kModes[] =
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

static void errorCallback(int error, const char* description)
{
    fputs(description, stderr);
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

int main(int argc, char** argv)
{
    int windowWidth = 1280, windowHeight = 1024;

    std::unique_ptr<WorkQueue> queue(new WorkQueue(WorkQueue::getIdealWorkerCount()));

    PhysSystem physSystem;

    RigidBody* groundBody = physSystem.AddBody(Coords2f(Vector2f(0, 0), 0.0f), Vector2f(10000.f, 10.0f));
    groundBody->invInertia = 0.0f;
    groundBody->invMass = 0.0f;

    int currentMode = sizeof(kModes) / sizeof(kModes[0]) - 1;

    const float gravity = -200.0f;
    const float integrationTime = 1 / 60.f;
    const int contactIterationsCount = 15;
    const int penetrationIterationsCount = 15;

    physSystem.gravity = gravity;

    float physicsTime = 0.0f;

    RigidBody* draggedBody = physSystem.AddBody(
        Coords2f(Vector2f(-500, 500), 0.0f), Vector2f(30.0f, 30.0f));

    float bodyRadius = 2.f;
    int bodyCount = 10000;

    for (int bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++)
    {
        Vector2f pos = Vector2f(random(-250.0f, 250.0f), random(50.f, 1000.0f));
        Vector2f size(bodyRadius * 2.f, bodyRadius * 2.f);

        physSystem.AddBody(Coords2f(pos, 0.f), size);
    }

    if (argc > 1 && strcmp(argv[1], "profile") == 0)
    {
        for (int mode = 0; mode < sizeof(kModes) / sizeof(kModes[0]); ++mode)
        {
            PhysSystem testSystem;

            testSystem.gravity = gravity;

            for (auto& body: physSystem.bodies)
            {
                RigidBody* testBody = testSystem.AddBody(body.coords, body.geom.size);
                testBody->invInertia = body.invInertia;
                testBody->invMass = body.invMass;
            }

            double collisionTime = 0;
            double mergeTime = 0;
            double solveTime = 0;
            float iterations = 0;

            for (int i = 0; i < 50; ++i)
            {
                testSystem.Update(*queue, 1.f / 30.f, kModes[mode].mode, contactIterationsCount, penetrationIterationsCount);

                collisionTime += testSystem.collisionTime;
                mergeTime += testSystem.mergeTime;
                solveTime += testSystem.solveTime;
                iterations += testSystem.iterations;
            }

            printf("%s: collision %.2f ms, merge %.2f ms, solve %.2f ms, %.2f iterations\n", kModes[mode].name, collisionTime * 1000, mergeTime * 1000, solveTime * 1000, iterations);
        }

        return 0;
    }

    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) return 1;

    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "PhyX", NULL, NULL);
    if (!window) return 1;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    glfwSetKeyCallback(window, keyCallback);

    double prevUpdateTime = 0.0f;

    bool paused = false;

    std::vector<Vertex> vertices;

    while (!glfwWindowShouldClose(window))
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        float ratio = float(width) / float(height);

        glViewport(0, 0, width, height);
        glClearColor(0.2f, 0.2f, 0.2f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);

        float offsetx = -width / 2;
        float offsety = -40;

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(offsetx, width + offsetx, offsety, height + offsety, 1.f, -1.f);

        vertices.clear();

        if (glfwGetTime() > prevUpdateTime + integrationTime)
        {
            prevUpdateTime += integrationTime;

            // float time = (paused && !sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) ? 0 : integrationTime;
            float time = integrationTime;

            RigidBody* draggedBody = &physSystem.bodies[1];
            // Vector2f dstVelocity = (mousePos - draggedBody->coords.pos) * 5e1f;
            // draggedBody->acceleration += (dstVelocity - draggedBody->velocity) * 5e0;

            physSystem.Update(*queue, time, kModes[currentMode].mode, contactIterationsCount, penetrationIterationsCount);
        }

        for (size_t bodyIndex = 0; bodyIndex < physSystem.bodies.size(); bodyIndex++)
        {
            RigidBody* body = &physSystem.bodies[bodyIndex];
            Coords2f bodyCoords = body->coords;
            Vector2f size = body->geom.size;

            float colorMult = float(bodyIndex) / float(physSystem.bodies.size()) * 0.5f + 0.5f;
            int r = 50 * colorMult;
            int g = 125 * colorMult;
            int b = 218 * colorMult;

            if (bodyIndex == 1) //dragged body
            {
                r = 242;
                g = 236;
                b = 164;
            }

            RenderBox(vertices, bodyCoords, size, r, g, b);
        }

        /*
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::V))
            for (size_t manifoldIndex = 0; manifoldIndex < physSystem.collider.manifolds.size(); manifoldIndex++)
            {
                Manifold& man = physSystem.collider.manifolds[manifoldIndex];

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

        window->draw(text2);

        window->display();
        */

        if (!vertices.empty())
        {
            glEnableClientState(GL_VERTEX_ARRAY);
            glEnableClientState(GL_COLOR_ARRAY);

            glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(Vertex), &vertices[0].r);
            glVertexPointer(2, GL_FLOAT, sizeof(Vertex), &vertices[0].position);
            glDrawArrays(GL_QUADS, 0, vertices.size());

            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableClientState(GL_COLOR_ARRAY);
        }

        /*
            << "Bodies: " << physSystem.bodies.size()
            << " Contacts: " << physSystem.solver.contactJoints.size()
            << " Iterations: " << physSystem.iterations;

            << queue->getWorkerCount() << " cores; "
            << "Mode: " << kModes[currentMode].name << "; "
            << "Physics time: " << std::setw(5) << physicsTime * 1000.0f << "ms (c: " << std::setw(5) << physSystem.collisionTime * 1000.0f << "ms, m: " << std::setw(5) << physSystem.mergeTime * 1000.0f << "ms, s: " << std::setw(5) << physSystem.solveTime * 1000.0f << "ms)";
        */

        glfwSwapBuffers(window);
        glfwPollEvents();

        /*
        while (window->pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window->close();
            else if (event.type == sf::Event::Resized)
            {
                sf::Vector2u size = window->getSize();

                window->setView(sf::View(sf::FloatRect(0.f, 0.f, static_cast<float>(size.x), static_cast<float>(size.y))));
            }
            else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::M)
                currentMode = (currentMode + 1) % (sizeof(kModes) / sizeof(kModes[0]));
            else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::P)
                paused = !paused;
            else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::C)
            {
                size_t workers =
                    (queue->getWorkerCount() == WorkQueue::getIdealWorkerCount())
                    ? 1
                    : std::min(queue->getWorkerCount() * 2, WorkQueue::getIdealWorkerCount());

                queue.reset(new WorkQueue(workers));
            }
        }
        */
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}