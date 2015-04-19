#pragma once

#include <immintrin.h>

#ifdef _MSC_VER
#define SIMD_INLINE __forceinline
#else
#define SIMD_INLINE __attribute__((always_inline))
#endif

#ifdef __SSE2__
#include "SIMD_SSE2.h"
#endif

#ifdef __AVX2__
#include "SIMD_AVX2.h"
#endif