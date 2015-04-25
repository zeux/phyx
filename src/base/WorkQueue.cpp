#include "WorkQueue.h"

#include "microprofile.h"

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
    pushItem(std::shared_ptr<Item>(), workers.size());

    for (unsigned int i = 0; i < workers.size(); ++i)
        workers[i].join();
}

void WorkQueue::pushItem(std::shared_ptr<Item> item, int count)
{
    std::unique_lock<std::mutex> lock(itemsMutex);

    items.push(std::make_pair(std::move(item), count));

    lock.unlock();

    if (count > 1)
        itemsCondition.notify_all();
    else
        itemsCondition.notify_one();
}

void WorkQueue::workerThreadFun(WorkQueue* queue, int worker)
{
    char name[16];
    sprintf(name, "Worker %d", worker);
    MicroProfileOnThreadCreate(name);

    for (;;)
    {
        std::shared_ptr<Item> item;

        {
            std::unique_lock<std::mutex> lock(queue->itemsMutex);

            queue->itemsCondition.wait(lock, [&]() { return !queue->items.empty(); });

            auto& slot = queue->items.front();

            if (slot.second <= 1)
            {
                item = std::move(slot.first);
                queue->items.pop();
            }
            else
            {
                item = slot.first;
                slot.second--;
            }
        }

        if (!item) break;

        MICROPROFILE_SCOPEI("WorkQueue", "JobRun", 0x606060);

        item->run(worker);
    }

    MicroProfileOnThreadExit();
}