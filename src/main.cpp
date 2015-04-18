#include <GLFW/glfw3.h>

#include "PhysSystem.h"

#include "microprofile/microprofile.h"
#include "microprofile/microprofileui.h"

struct Vertex
{
    Vector2f position;
    unsigned char r, g, b, a;

};

void RenderBox(std::vector<Vertex>& vertices, Coords2f coords, Vector2f size, int r, int g, int b, int a)
{
    Vector2f axisX = coords.xVector * size.x;
    Vector2f axisY = coords.yVector * size.y;

    Vertex v;

    v.r = r;
    v.g = g;
    v.b = b;
    v.a = a;

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

bool keyPressed[GLFW_KEY_LAST + 1];
int mouseScrollDelta = 0;

static void errorCallback(int error, const char* description)
{
    fputs(description, stderr);
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    keyPressed[key] = (action == GLFW_PRESS);
}

static void scrollCallback(GLFWwindow* window, double x, double y)
{
    mouseScrollDelta = y;
}

void MicroProfileDrawInit();
void MicroProfileBeginDraw();
void MicroProfileEndDraw();

int main(int argc, char** argv)
{
    MicroProfileOnThreadCreate("Main");

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

    physSystem.AddBody(Coords2f(Vector2f(-500, 500), 0.0f), Vector2f(30.0f, 30.0f));

    float bodyRadius = 2.f;
    int bodyCount = 20000;

    for (int bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++)
    {
        Vector2f pos = Vector2f(random(-500.0f, 500.0f), random(50.f, 1000.0f));
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
    glfwSetScrollCallback(window, scrollCallback);

    MicroProfileDrawInit();

    double prevUpdateTime = 0.0f;

    std::vector<Vertex> vertices;

    while (!glfwWindowShouldClose(window))
    {
        MICROPROFILE_SCOPEI("MAIN", "Frame", 0xffee00);

        int width, height;
        glfwGetWindowSize(window, &width, &height);

        int frameWidth, frameHeight;
        glfwGetFramebufferSize(window, &frameWidth, &frameHeight);

        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);

        glViewport(0, 0, frameWidth, frameHeight);
        glClearColor(0.2f, 0.2f, 0.2f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);

        float offsetx = -width / 2;
        float offsety = -40;
        float scale = 0.5f;

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(offsetx / scale, width / scale + offsetx / scale, offsety / scale, height / scale + offsety / scale, 1.f, -1.f);

        vertices.clear();

        if (glfwGetTime() > prevUpdateTime + integrationTime)
        {
            prevUpdateTime += integrationTime;

            float time = integrationTime;

            Vector2f mousePos = Vector2f(mouseX + offsetx, height + offsety - mouseY) / scale;

            RigidBody* draggedBody = &physSystem.bodies[1];
            Vector2f dstVelocity = (mousePos - draggedBody->coords.pos) * 5e1f;
            draggedBody->acceleration += (dstVelocity - draggedBody->velocity) * 5e0;

            physSystem.Update(*queue, time, kModes[currentMode].mode, contactIterationsCount, penetrationIterationsCount);
        }

        {
            MICROPROFILE_SCOPEI("Render", "Prepare", 0xff0000);

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

                RenderBox(vertices, bodyCoords, size, r, g, b, 255);
            }

            if (glfwGetKey(window, GLFW_KEY_V))
            {
                for (size_t manifoldIndex = 0; manifoldIndex < physSystem.collider.manifolds.size(); manifoldIndex++)
                {
                    Manifold& man = physSystem.collider.manifolds[manifoldIndex];

                    for (int collisionNumber = 0; collisionNumber < man.collisionsCount; collisionNumber++)
                    {
                        Coords2f coords = Coords2f(Vector2f(0.0f, 0.0f), 3.1415f / 4.0f);

                        coords.pos = man.body1->coords.pos + man.collisions[collisionNumber].delta1;

                        float redMult = man.collisions[collisionNumber].isNewlyCreated ? 0.5f : 1.0f;

                        RenderBox(vertices, coords, Vector2f(3.0f, 3.0f), 100, 100 * redMult, 100 * redMult, 100);

                        coords.pos = man.body2->coords.pos + man.collisions[collisionNumber].delta2;

                        RenderBox(vertices, coords, Vector2f(3.0f, 3.0f), 150, 150 * redMult, 150 * redMult, 100);
                    }
                }
            }
        }

        {
            MICROPROFILE_SCOPEI("Render", "Perform", 0xff0000);

            if (!vertices.empty())
            {
                glEnableClientState(GL_VERTEX_ARRAY);
                glEnableClientState(GL_COLOR_ARRAY);

                glVertexPointer(2, GL_FLOAT, sizeof(Vertex), &vertices[0].position);
                glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(Vertex), &vertices[0].r);

                glDrawArrays(GL_QUADS, 0, vertices.size());

                glDisableClientState(GL_VERTEX_ARRAY);
                glDisableClientState(GL_COLOR_ARRAY);
            }
        }

        char stats[256];
        sprintf(stats, "Bodies: %d Manifolds: %d Contacts: %d | Cores: %d; Mode: %s; Iterations: %.2f",
            int(physSystem.bodies.size()),
            int(physSystem.collider.manifolds.size()),
            int(physSystem.solver.contactJoints.size()),
            int(queue->getWorkerCount()),
            kModes[currentMode].name,
            physSystem.iterations);

        MicroProfileFlip();

        {
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0, width, height, 0, 1.f, -1.f);

            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDisable(GL_DEPTH_TEST);

            MicroProfileBeginDraw();

            MicroProfileDraw(width, height);
            MicroProfileDrawText(2, height - 12, 0xffffffff, stats, strlen(stats));

            MicroProfileEndDraw();

            glDisable(GL_BLEND);
        }

        {
            MICROPROFILE_SCOPEI("MAIN", "Flip", 0xffee00);

            glfwSwapBuffers(window);
        }

        {
            MICROPROFILE_SCOPEI("MAIN", "Input", 0xffee00);

            // Handle input
            memset(keyPressed, 0, sizeof(keyPressed));
            mouseScrollDelta = 0;

            glfwPollEvents();

            bool mouseDown0 = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
            bool mouseDown1 = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

            MicroProfileMouseButton(mouseDown0, mouseDown1);
            MicroProfileMousePosition(mouseX, mouseY, mouseScrollDelta);
            MicroProfileModKey(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS);

            if (keyPressed[GLFW_KEY_ESCAPE])
                break;

            if (keyPressed[GLFW_KEY_O])
                MicroProfileToggleDisplayMode();

            if (keyPressed[GLFW_KEY_P])
                MicroProfileTogglePause();

            if (keyPressed[GLFW_KEY_M])
                currentMode = (currentMode + 1) % (sizeof(kModes) / sizeof(kModes[0]));

            if (keyPressed[GLFW_KEY_C])
            {
                size_t workers =
                    (queue->getWorkerCount() == WorkQueue::getIdealWorkerCount())
                    ? 1
                    : std::min(queue->getWorkerCount() * 2, WorkQueue::getIdealWorkerCount());

                queue.reset(new WorkQueue(workers));
            }
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}