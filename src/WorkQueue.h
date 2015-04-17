#pragma once

#include <vector>
#include <thread>
#include <functional>

#include "BlockingQueue.h"

class WorkQueue
{
public:
	static unsigned int getIdealWorkerCount()
	{
		return std::max(std::thread::hardware_concurrency(), 1u);
	}

	WorkQueue(size_t workerCount)
	{
		for (size_t i = 0; i < workerCount; ++i)
			workers.emplace_back(std::bind(workerThreadFun, std::ref(queue)));
	}

	~WorkQueue()
	{
		for (size_t i = 0; i < workers.size(); ++i)
			queue.push(std::function<void()>());

		for (size_t i = 0; i < workers.size(); ++i)
			workers[i].join();
	}

	void push(std::function<void()> fun)
	{
		queue.push(std::move(fun));
	}

	size_t getWorkerCount() const
	{
		return workers.size();
	}

private:
	BlockingQueue<std::function<void()>> queue;
	std::vector<std::thread> workers;

	static void workerThreadFun(BlockingQueue<std::function<void()>>& queue)
	{
		while (std::function<void()> fun = queue.pop())
		{
			fun();
		}
	}
};