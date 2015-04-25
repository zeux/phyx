#pragma once

#include "WorkQueue.h"

#include <atomic>

#include "microprofile.h"

template <typename T>
inline T& parallelForIndex(T* data, unsigned int index)
{
    return data[index];
}

inline unsigned int parallelForIndex(unsigned int data, unsigned int index)
{
    return data + index;
}

template <typename T, typename F>
inline void serialFor(WorkQueue& queue, T data, unsigned int count, unsigned int groupSize, F func)
{
    for (unsigned int i = 0; i < count; ++i)
        func(parallelForIndex(data, i), 0);
}

template <typename T, typename F>
inline void parallelFor(WorkQueue& queue, T data, unsigned int count, unsigned int groupSize, F func)
{
    if (queue.getWorkerCount() == 0 || count <= groupSize)
    {
        for (unsigned int i = 0; i < count; ++i)
            func(parallelForIndex(data, i), 0);

        return;
    }

    MICROPROFILE_SCOPEI("WorkQueue", "ParallelFor", 0x808080);

    struct Item: WorkQueue::Item
    {
        WorkQueue* queue;

        std::atomic<unsigned int> counter;
        std::atomic<unsigned int> ready;

        T data;
        unsigned int count;

        unsigned int groupSize;
        unsigned int groupCount;

        F* func;

        Item(): counter(0), ready(0)
        {
        }

        void run(int worker) override
        {
            unsigned int groups = 0;

            for (;;)
            {
                unsigned int groupIndex = counter++;
                if (groupIndex >= groupCount) break;

                unsigned int begin = groupIndex * groupSize;
                unsigned int end = std::min(count, begin + groupSize);

                for (unsigned int i = begin; i < end; ++i)
                    (*func)(parallelForIndex(data, i), worker);

                groups++;
            }

            ready += groups;
        }
    };

    auto item = std::make_shared<Item>();

    item->queue = &queue;
    item->data = data;
    item->count = count;
    item->groupSize = groupSize;
    item->groupCount = (count + groupSize - 1) / groupSize;
    item->func = &func;

    int optimalWorkerCount = std::min(unsigned(queue.getWorkerCount()), item->groupCount - 1);

    {
        MICROPROFILE_SCOPEI("WorkQueue", "Push", 0x00ff00);

        queue.pushItem(item, optimalWorkerCount);
        //for (int i = 0; i < optimalWorkerCount; ++i)
            //queue.pushItem(item);
    }

    item->run(queue.getWorkerCount());

    if (item->ready.load() < item->groupCount)
    {
        MICROPROFILE_SCOPEI("WorkQueue", "Wait", 0xff0000);

        do std::this_thread::yield();
        while (item->ready.load() < item->groupCount);
    }
}