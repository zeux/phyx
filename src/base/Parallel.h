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
inline void parallelFor(WorkQueue& queue, T data, unsigned int count, unsigned int groupSize, F func)
{
    MICROPROFILE_SCOPEI("WorkQueue", "ParallelFor", 0x808080);

    if (queue.getWorkerCount() == 0)
    {
        for (unsigned int i = 0; i < count; ++i)
            func(parallelForIndex(data, i), 0);

        return;
    }

    struct State
    {
        WorkQueue* queue;

        std::atomic<unsigned int> counter;
        std::atomic<unsigned int> ready;

        T data;
        unsigned int count;

        unsigned int groupSize;
        unsigned int groupCount;

        F* func;

        State(): counter(0), ready(0)
        {
        }
    };

    struct Item: WorkQueue::Item
    {
        std::shared_ptr<State> state;

        Item(const std::shared_ptr<State>& state): state(state)
        {
        }

        void run(int worker) override
        {
            unsigned int groups = 0;

            for (;;)
            {
                unsigned int groupIndex = state->counter++;
                if (groupIndex >= state->groupCount) break;

                unsigned int begin = groupIndex * state->groupSize;
                unsigned int end = std::min(state->count, begin + state->groupSize);

                for (unsigned int i = begin; i < end; ++i)
                    (*state->func)(parallelForIndex(state->data, i), worker);

                groups++;
            }

            state->ready += groups;
        }
    };

    auto state = std::make_shared<State>();

    state->queue = &queue;
    state->data = data;
    state->count = count;
    state->groupSize = groupSize;
    state->groupCount = (count + groupSize - 1) / groupSize;
    state->func = &func;

    std::vector<std::unique_ptr<WorkQueue::Item>> items(queue.getWorkerCount());

    for (size_t i = 0; i < queue.getWorkerCount(); ++i)
        items[i] = std::unique_ptr<WorkQueue::Item>(new Item(state));

    {
        MICROPROFILE_SCOPEI("WorkQueue", "Push", 0x00ff00);

        queue.push(std::move(items));
    }

    Item item(state);
    item.run(queue.getWorkerCount());

    if (state->ready.load() < state->groupCount)
    {
        MICROPROFILE_SCOPEI("WorkQueue", "Wait", 0xff0000);

        do std::this_thread::yield();
        while (state->ready.load() < state->groupCount);
    }
}