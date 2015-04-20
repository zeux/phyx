#include <GLFW/glfw3.h>

#include "World.h"

#include "base/WorkQueue.h"

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
    World::SolveMode mode;
    const char* name;
} kModes[] =
    {
     {World::Solve_AoS, "AoS"},
     {World::Solve_SoA_Scalar, "SoA Scalar"},
     {World::Solve_SoA_SSE2, "SoA SSE2"},

#ifdef __AVX2__
     {World::Solve_SoA_AVX2, "SoA AVX2"},
#endif
};

const char* resetWorld(World& world, int scene)
{
    world.bodies.clear();
    world.collider.manifolds.clear();
    world.collider.manifoldMap.clear();
    world.solver.contactJoints.clear();

    RigidBody* groundBody = world.AddBody(Coords2f(Vector2f(0, 0), 0.0f), Vector2f(10000.f, 10.0f));
    groundBody->invInertia = 0.0f;
    groundBody->invMass = 0.0f;

    world.AddBody(Coords2f(Vector2f(-1000, 1000), 0.0f), Vector2f(30.0f, 30.0f));

    switch (scene % 2)
    {
    case 0:
    {
        for (int bodyIndex = 0; bodyIndex < 20000; bodyIndex++)
        {
            Vector2f pos = Vector2f(random(-500.0f, 500.0f), random(50.f, 1000.0f));
            Vector2f size(4.f, 4.f);

            world.AddBody(Coords2f(pos, 0.f), size);
        }

        return "Falling";
    }

    case 1:
    {
        for (int left = -100; left <= 100; left++)
        {
            for (int bodyIndex = 0; bodyIndex < 100; bodyIndex++)
            {
                Vector2f pos = Vector2f(left * 20, 10 + bodyIndex * 10);
                Vector2f size(10, 5);

                world.AddBody(Coords2f(pos, 0.f), size);
            }
        }

        return "Wall";
    }
    }

    return "Empty";
}

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

    World world;

    int currentMode = sizeof(kModes) / sizeof(kModes[0]) - 1;
    int currentScene = 0;

    const char* currentSceneName = resetWorld(world, currentScene);

    const float gravity = -200.0f;
    const float integrationTime = 1 / 60.f;
    const int contactIterationsCount = 15;
    const int penetrationIterationsCount = 15;

    world.gravity = gravity;

    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) return 1;

    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "PhyX", NULL, NULL);
    if (!window) return 1;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetScrollCallback(window, scrollCallback);

    MicroProfileDrawInit();

    bool paused = false;

    double prevUpdateTime = 0.0f;

    std::vector<Vertex> vertices;

    float viewOffsetX = -500;
    float viewOffsetY = -40;
    float viewScale = 0.5f;

    while (!glfwWindowShouldClose(window))
    {
        MicroProfileFlip();

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

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(viewOffsetX / viewScale, width / viewScale + viewOffsetX / viewScale, viewOffsetY / viewScale, height / viewScale + viewOffsetY / viewScale, 1.f, -1.f);

        vertices.clear();

        if (glfwGetTime() > prevUpdateTime + integrationTime)
        {
            prevUpdateTime += integrationTime;

            if (!paused)
            {
                Vector2f mousePos = Vector2f(mouseX + viewOffsetX, height + viewOffsetY - mouseY) / viewScale;

                RigidBody* draggedBody = &world.bodies[1];
                Vector2f dstVelocity = (mousePos - draggedBody->coords.pos) * 5e1f;
                draggedBody->acceleration += (dstVelocity - draggedBody->velocity) * 5e0;

                world.Update(*queue, integrationTime, kModes[currentMode].mode, contactIterationsCount, penetrationIterationsCount);
            }
        }

        char stats[256];
        sprintf(stats, "Scene: %s | Bodies: %d Manifolds: %d Contacts: %d | Cores: %d; Mode: %s; Iterations: %.2f",
            currentSceneName,
            int(world.bodies.size()),
            int(world.collider.manifolds.size()),
            int(world.solver.contactJoints.size()),
            int(queue->getWorkerCount()),
            kModes[currentMode].name,
            world.iterations);

        {
            MICROPROFILE_SCOPEI("Render", "Render", 0xff0000);

            {
                MICROPROFILE_SCOPEI("Render", "Prepare", -1);

                for (size_t bodyIndex = 0; bodyIndex < world.bodies.size(); bodyIndex++)
                {
                    RigidBody* body = &world.bodies[bodyIndex];
                    Coords2f bodyCoords = body->coords;
                    Vector2f size = body->geom.size;

                    float colorMult = float(bodyIndex) / float(world.bodies.size()) * 0.5f + 0.5f;
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
                    for (size_t manifoldIndex = 0; manifoldIndex < world.collider.manifolds.size(); manifoldIndex++)
                    {
                        Manifold& man = world.collider.manifolds[manifoldIndex];

                        for (int collisionNumber = 0; collisionNumber < man.pointCount; collisionNumber++)
                        {
                            Coords2f coords = Coords2f(Vector2f(0.0f, 0.0f), 3.1415f / 4.0f);

                            coords.pos = man.body1->coords.pos + man.points[collisionNumber].delta1;

                            float redMult = man.points[collisionNumber].isNewlyCreated ? 0.5f : 1.0f;

                            RenderBox(vertices, coords, Vector2f(3.0f, 3.0f), 100, 100 * redMult, 100 * redMult, 100);

                            coords.pos = man.body2->coords.pos + man.points[collisionNumber].delta2;

                            RenderBox(vertices, coords, Vector2f(3.0f, 3.0f), 150, 150 * redMult, 150 * redMult, 100);
                        }
                    }
                }
            }

            {
                MICROPROFILE_SCOPEI("Render", "Perform", -1);

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

            {
                MICROPROFILE_SCOPEI("Render", "Profile", -1);

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
            {
                paused = !paused;
                MicroProfileTogglePause();
            }

            if (keyPressed[GLFW_KEY_M])
                currentMode = (currentMode + 1) % (sizeof(kModes) / sizeof(kModes[0]));

            if (keyPressed[GLFW_KEY_R])
                currentSceneName = resetWorld(world, currentScene);

            if (keyPressed[GLFW_KEY_S])
                currentSceneName = resetWorld(world, ++currentScene);

            if (keyPressed[GLFW_KEY_C])
            {
                size_t workers =
                    (queue->getWorkerCount() == WorkQueue::getIdealWorkerCount())
                    ? 1
                    : std::min(queue->getWorkerCount() * 2, WorkQueue::getIdealWorkerCount());

                queue.reset(new WorkQueue(workers));
            }

            if (glfwGetKey(window, GLFW_KEY_LEFT))
                viewOffsetX -= 1;

            if (glfwGetKey(window, GLFW_KEY_RIGHT))
                viewOffsetX += 1;
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}