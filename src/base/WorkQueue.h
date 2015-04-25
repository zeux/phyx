#pragma once

#include <vector>
#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>

class WorkQueue
{
  public:
    struct Item
    {
        virtual ~Item() {}

        virtual void run(int worker) = 0;
    };

    static unsigned int getIdealWorkerCount();

    WorkQueue(unsigned int workerCount);
    ~WorkQueue();

    void push(std::unique_ptr<Item> item);
    void push(std::vector<std::unique_ptr<Item>> item);
    void push(std::function<void()> fun);

    unsigned int getWorkerCount() const
    {
        return workers.size();
    }

  private:
    std::vector<std::thread> workers;

    std::mutex itemsMutex;
    std::condition_variable itemsCondition;
    std::queue<std::unique_ptr<Item>> items;

    static void workerThreadFun(WorkQueue* queue, int worker);
};