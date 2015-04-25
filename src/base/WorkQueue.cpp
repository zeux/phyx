#include "WorkQueue.h"

#include "microprofile.h"

struct WorkItemFunction: WorkQueue::Item
{
    std::function<void()> function;

    void run(int worker) override
    {
        function();
    }
};

unsigned int WorkQueue::getIdealWorkerCount()
{
    return std::max(std::thread::hardware_concurrency(), 1u);
}

WorkQueue::WorkQueue(unsigned int workerCount)
{
    for (unsigned int i = 0; i < workerCount; ++i)
        workers.emplace_back(workerThreadFun, this, i);
}

WorkQueue::~WorkQueue()
{
    for (unsigned int i = 0; i < workers.size(); ++i)
        push(std::unique_ptr<Item>());

    for (unsigned int i = 0; i < workers.size(); ++i)
        workers[i].join();
}

void WorkQueue::push(std::unique_ptr<Item> item)
{
    std::unique_lock<std::mutex> lock(itemsMutex);

    items.push(std::move(item));

    lock.unlock();
    itemsCondition.notify_one();
}

void WorkQueue::push(std::vector<std::unique_ptr<Item>> item)
{
    std::unique_lock<std::mutex> lock(itemsMutex);

    for (auto&& i: item)
        items.push(std::move(i));

    lock.unlock();
    itemsCondition.notify_all();
}

void WorkQueue::push(std::function<void()> fun)
{
    std::unique_ptr<WorkItemFunction> item(new WorkItemFunction());
    item->function = std::move(fun);

    push(std::unique_ptr<Item>(std::move(item)));
}

void WorkQueue::workerThreadFun(WorkQueue* queue, int worker)
{
    char name[16];
    sprintf(name, "Worker %d", worker);
    MicroProfileOnThreadCreate(name);

    for (;;)
    {
        std::unique_ptr<Item> item;

        {
            std::unique_lock<std::mutex> lock(queue->itemsMutex);

            queue->itemsCondition.wait(lock, [&]() { return !queue->items.empty(); });

            item = std::move(queue->items.front());
            queue->items.pop();
        }

        if (!item) break;

        MICROPROFILE_SCOPEI("WorkQueue", "JobRun", 0x606060);

        item->run(worker);
    }

    MicroProfileOnThreadExit();
}