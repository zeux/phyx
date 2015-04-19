#pragma once

#include <xmmintrin.h>

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

    T& operator[](int i)
    {
        return data[i];
    }

    void resize(int newsize)
    {
        if (newsize > capacity)
        {
            int newcapacity = capacity;
            while (newcapacity < newsize)
                newcapacity += newcapacity / 2 + 1;

            _mm_free(data);

            data = static_cast<T*>(_mm_malloc(newcapacity * sizeof(T), 32));
            capacity = newcapacity;
        }

        size = newsize;
    }
};