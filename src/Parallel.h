#pragma once

#include "WorkQueue.h"

#include <atomic>

template <typename T, typename F>
inline void ParallelFor(WorkQueue& queue, T* data, unsigned int count, unsigned int groupSize, F func)
{
    if (queue.getWorkerCount() == 1)
    {
        for (unsigned int i = 0; i < count; ++i)
            func(data[i], 0);

        return;
    }

    struct Item: WorkQueue::Item
    {
        WorkQueue* queue;

        std::atomic<unsigned int>* counter;
        std::atomic<unsigned int>* ready;

        T* data;
        unsigned int count;
        unsigned int groupSize;

        F* func;

        void run(int worker) override
        {
            for (;;)
            {
                unsigned int index = (*counter)++ * groupSize;
                if (index >= count) break;

                unsigned int end = std::min(count, index + groupSize);

                for (unsigned int i = index; i < end; ++i)
                    (*func)(data[i], worker);
            }

            if (++*ready == queue->getWorkerCount())
                queue->signalReady();
        }
    };

    std::atomic<unsigned int> counter(0);
    std::atomic<unsigned int> ready(0);

    for (size_t i = 0; i < queue.getWorkerCount(); ++i)
    {
        std::unique_ptr<Item> item(new Item());
        item->queue = &queue;
        item->counter = &counter;
        item->ready = &ready;
        item->data = data;
        item->count = count;
        item->groupSize = groupSize;
        item->func = &func;

        queue.push(std::unique_ptr<WorkQueue::Item>(std::move(item)));
    }

    queue.signalWait();
}