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

    void pushItem(std::shared_ptr<Item> item, int count = 1);

    template <typename F>
    void pushFunction(F fun, int count = 1)
    {
        pushItem(std::make_shared<Item>(std::move(fun)));
    }

    unsigned int getWorkerCount() const
    {
        return workers.size();
    }

  private:
    std::vector<std::thread> workers;

    std::mutex itemsMutex;
    std::condition_variable itemsCondition;
    std::queue<std::pair<std::shared_ptr<Item>, int>> items;

    static void workerThreadFun(WorkQueue* queue, int worker);

    template <typename T>
    struct ItemFunction: WorkQueue::Item
    {
        T function;

        ItemFunction(T function): function(std::move(function))
        {
        }

        void run(int worker) override
        {
            function();
        }
    };

};