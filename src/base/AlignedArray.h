#pragma once

#include <xmmintrin.h>
#include <string.h>

#ifdef _MSC_VER
#define _mm_malloc _aligned_malloc
#define _mm_free _aligned_free
#endif

template <typename T>
struct AlignedArray
{
    T* data;
    int size;
    int capacity;

    AlignedArray()
        : data(0)
        , size(0)
        , capacity(0)
    {
    }

    ~AlignedArray()
    {
        _mm_free(data);
    }

    AlignedArray(const AlignedArray&) = delete;
    AlignedArray& operator=(const AlignedArray&) = delete;

    AlignedArray(AlignedArray&& other)
    {
        data = other.data;
        size = other.size;
        capacity = other.capacity;

        other.data = 0;
        other.size = 0;
        other.capacity = 0;
    }

    AlignedArray& operator=(AlignedArray&& other)
    {
        _mm_free(data);

        data = other.data;
        size = other.size;
        capacity = other.capacity;

        other.data = 0;
        other.size = 0;
        other.capacity = 0;
    }

    T* begin()
    {
        return data;
    }

    T* end()
    {
        return data + size;
    }

    T& operator[](int i)
    {
        assert(i >= 0 && i < size);
        return data[i];
    }

    void push_back(const T& value)
    {
        if (size == capacity)
        {
            T copy = value;

            realloc(size + 1, true);

            data[size++] = copy;
        }
        else
        {
            data[size++] = value;
        }
    }

    void truncate(int newsize)
    {
        assert(newsize <= size);

        size = newsize;
    }

    void resize_copy(int newsize)
    {
        if (newsize > capacity)
            realloc(newsize, true);

        size = newsize;
    }

    void resize(int newsize)
    {
        if (newsize > capacity)
            realloc(newsize, false);

        size = newsize;
    }

    void realloc(int newsize, bool copy)
    {
        int newcapacity = capacity;
        while (newcapacity < newsize)
            newcapacity += newcapacity / 2 + 1;

        // Leave 32b padding at the end to avoid buffer overruns for fast SIMD code
        T* newdata = static_cast<T*>(_mm_malloc(newcapacity * sizeof(T) + 32, 32));

        if (data)
        {
            if (copy)
                memcpy(newdata, data, size * sizeof(T));
            _mm_free(data);
        }

        data = newdata;
        capacity = newcapacity;
    }

    void clear()
    {
        size = 0;
    }
};
