#pragma once

#include <immintrin.h>

#ifdef _MSC_VER
#define SIMD_INLINE __forceinline
#else
#define SIMD_INLINE __attribute__((always_inline))
#endif

namespace simd
{
	template <int N> struct VNf_;
	template <int N> struct VNi_;
	template <int N> struct VNb_;

	template <int N> using VNf = typename VNf_<N>::type;
	template <int N> using VNi = typename VNi_<N>::type;
	template <int N> using VNb = typename VNb_<N>::type;

	template <typename T> void dump(const char* name, const T& v)
	{
		printf("%s:", name);

		const float* fptr = reinterpret_cast<const float*>(&v);
		const int* iptr = reinterpret_cast<const int*>(&v);

		for (size_t i = 0; i < sizeof(v) / 4; ++i)
			printf(" %f [%08x]", fptr[i], iptr[i]);

		printf("\n");
	}
}

#define SIMD_DUMP(v) simd::dump(#v, v)

#include "SIMD_Scalar.h"

#ifdef __SSE2__
#include "SIMD_SSE2.h"
#endif

#ifdef __AVX2__
#include "SIMD_AVX2.h"
#endif