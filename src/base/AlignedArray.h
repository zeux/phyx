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

            // Leave 32b padding at the end to avoid buffer overruns for fast SIMD code
            T* newdata = static_cast<T*>(_mm_malloc(newcapacity * sizeof(T) + 32, 32));

            if (data)
            {
                memcpy(newdata, data, size * sizeof(T));
                _mm_free(data);
            }

            data = newdata;
            capacity = newcapacity;
        }

        size = newsize;
    }
};