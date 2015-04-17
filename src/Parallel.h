#pragma once

#include "WorkQueue.h"

template <typename T, typename F> inline NOINLINE void ParallelForBatch(T* data, unsigned int count, unsigned int groupSize, std::atomic<unsigned int>& counter, F& func)
{
	do
	{
		unsigned int index = (counter++) * groupSize;
		if (index >= count) return;

		unsigned int end = std::min(count, index + groupSize);

		for (unsigned int i = index; i < end; ++i)
			func(data[i]);
	}
	while (true);
}

template <typename T, typename F> inline void ParallelFor(WorkQueue& queue, T* data, unsigned int count, unsigned int groupSize, F func)
{
	std::atomic<unsigned int> counter(0);
	std::atomic<unsigned int> ready(0);

	for (size_t i = 0; i < queue.getWorkerCount(); ++i)
	{
		queue.push([&, data, count, groupSize]() {
			ParallelForBatch(data, count, groupSize, counter, func);

			if (++ready == queue.getWorkerCount())
				queue.signalReady();
		});
	}

	queue.signalWait();
}