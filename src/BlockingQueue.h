#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <cassert>

template <typename T>
class BlockingQueue
{
public:
	BlockingQueue()
	{
	}

	void push(T&& item)
	{
		std::unique_lock<std::mutex> lock(mutex);

		items.push(std::move(item));

		lock.unlock();
		itemsNotEmpty.notify_one();
	}

	T pop()
	{
		std::unique_lock<std::mutex> lock(mutex);

		itemsNotEmpty.wait(lock, [&]() { return !items.empty(); });

		T item = std::move(items.front());
		items.pop();

		return std::move(item);
	}

private:
	std::mutex mutex;
	std::condition_variable itemsNotEmpty;

	std::queue<T> items;
};
