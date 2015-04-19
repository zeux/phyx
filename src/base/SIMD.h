#pragma once

#include <immintrin.h>

#ifdef _MSC_VER
#define SIMD_INLINE __forceinline
#else
#define SIMD_INLINE __attribute__((always_inline))
#endif

namespace simd
{
	struct V4f
	{
		__m128 v;

		SIMD_INLINE V4f()
		{
		}

		SIMD_INLINE V4f(__m128 v): v(v)
		{
		}

		SIMD_INLINE operator __m128() const
		{
			return v;
		}

		static V4f zero()
		{
			return _mm_setzero_ps();
		}

		static V4f one(float v)
		{
			return _mm_set1_ps(v);
		}

		static V4f sign()
		{
			return _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
		}
	};

	struct V4i
	{
		__m128i v;

		SIMD_INLINE V4i()
		{
		}

		SIMD_INLINE V4i(__m128i v): v(v)
		{
		}

		SIMD_INLINE operator __m128i() const
		{
			return v;
		}

		static V4i zero()
		{
			return _mm_setzero_si128();
		}

		static V4i one(int v)
		{
			return _mm_set1_epi32(v);
		}
	};

	struct V8f
	{
		__m256 v;

		SIMD_INLINE V8f()
		{
		}

		SIMD_INLINE V8f(__m256 v): v(v)
		{
		}

		SIMD_INLINE operator __m256() const
		{
			return v;
		}

		static V8f zero()
		{
			return _mm256_setzero_ps();
		}

		static V8f one(float v)
		{
			return _mm256_set1_ps(v);
		}

		static V8f sign()
		{
			return _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
		}
	};

	struct V8i
	{
		__m256i v;

		SIMD_INLINE V8i()
		{
		}

		SIMD_INLINE V8i(__m256i v): v(v)
		{
		}

		SIMD_INLINE operator __m256i() const
		{
			return v;
		}

		static V8i zero()
		{
			return _mm256_setzero_si256();
		}

		static V8i one(int v)
		{
			return _mm256_set1_epi32(v);
		}
	};

	SIMD_INLINE V4f bitcast(V4i v)
	{
		return _mm_castsi128_ps(v.v);
	}

	SIMD_INLINE V4i bitcast(V4f v)
	{
		return _mm_castps_si128(v.v);
	}

	SIMD_INLINE V8i bitcast(V8f v)
	{
		return _mm256_castps_si256(v.v);
	}

	SIMD_INLINE V8f bitcast(V8i v)
	{
		return _mm256_castsi256_ps(v.v);
	}

	SIMD_INLINE V4f operator+(V4f v)
	{
		return v;
	}

	SIMD_INLINE V8f operator+(V8f v)
	{
		return v;
	}

	SIMD_INLINE V4f operator-(V4f v)
	{
		return _mm_xor_ps(V4f::sign(), v.v);
	}

	SIMD_INLINE V8f operator-(V8f v)
	{
		return _mm256_xor_ps(V8f::sign(), v.v);
	}

	SIMD_INLINE V4f operator+(V4f l, V4f r)
	{
		return _mm_add_ps(l.v, r.v);
	}

	SIMD_INLINE V8f operator+(V8f l, V8f r)
	{
		return _mm256_add_ps(l.v, r.v);
	}

	SIMD_INLINE V4f operator-(V4f l, V4f r)
	{
		return _mm_sub_ps(l.v, r.v);
	}

	SIMD_INLINE V8f operator-(V8f l, V8f r)
	{
		return _mm256_sub_ps(l.v, r.v);
	}

	SIMD_INLINE V4f operator*(V4f l, V4f r)
	{
		return _mm_mul_ps(l.v, r.v);
	}

	SIMD_INLINE V8f operator*(V8f l, V8f r)
	{
		return _mm256_mul_ps(l.v, r.v);
	}

	SIMD_INLINE V4f operator/(V4f l, V4f r)
	{
		return _mm_div_ps(l.v, r.v);
	}

	SIMD_INLINE V8f operator/(V8f l, V8f r)
	{
		return _mm256_div_ps(l.v, r.v);
	}

	SIMD_INLINE void operator+=(V4f& l, V4f r)
	{
		l.v = _mm_add_ps(l.v, r.v);
	}

	SIMD_INLINE void operator+=(V8f& l, V8f r)
	{
		l.v = _mm256_add_ps(l.v, r.v);
	}

	SIMD_INLINE void operator-=(V4f& l, V4f r)
	{
		l.v = _mm_sub_ps(l.v, r.v);
	}

	SIMD_INLINE void operator-=(V8f& l, V8f r)
	{
		l.v = _mm256_sub_ps(l.v, r.v);
	}

	SIMD_INLINE void operator*=(V4f& l, V4f r)
	{
		l.v = _mm_mul_ps(l.v, r.v);
	}

	SIMD_INLINE void operator*=(V8f& l, V8f r)
	{
		l.v = _mm256_mul_ps(l.v, r.v);
	}

	SIMD_INLINE void operator/=(V4f& l, V4f r)
	{
		l.v = _mm_div_ps(l.v, r.v);
	}

	SIMD_INLINE void operator/=(V8f& l, V8f r)
	{
		l.v = _mm256_div_ps(l.v, r.v);
	}

	SIMD_INLINE V4f abs(V4f v)
	{
		return _mm_andnot_ps(V4f::sign(), v.v);
	}

	SIMD_INLINE V8f abs(V8f v)
	{
		return _mm256_andnot_ps(V8f::sign(), v.v);
	}

	SIMD_INLINE V4f copysign(V4f x, V4f y)
	{
		V4f sign = V4f::sign();

		return _mm_or_ps(_mm_andnot_ps(sign.v, x.v), _mm_and_ps(y.v, sign.v));
	}

	SIMD_INLINE V8f copysign(V8f x, V8f y)
	{
		V8f sign = V8f::sign();

		return _mm256_or_ps(_mm256_andnot_ps(sign.v, x.v), _mm256_and_ps(y.v, sign.v));
	}

	SIMD_INLINE V4f flipsign(V4f x, V4f y)
	{
		return _mm_xor_ps(x.v, _mm_and_ps(y.v, V4f::sign()));
	}

	SIMD_INLINE V8f flipsign(V8f x, V8f y)
	{
		return _mm256_xor_ps(x.v, _mm256_and_ps(y.v, V8f::sign()));
	}

	SIMD_INLINE V4f min(V4f l, V4f r)
	{
		return _mm_min_ps(l.v, r.v);
	}

	SIMD_INLINE V8f min(V8f l, V8f r)
	{
		return _mm256_min_ps(l.v, r.v);
	}

	SIMD_INLINE V4f max(V4f l, V4f r)
	{
		return _mm_max_ps(l.v, r.v);
	}

	SIMD_INLINE V8f max(V8f l, V8f r)
	{
		return _mm256_max_ps(l.v, r.v);
	}
}

using simd::V4f;
using simd::V4i;
using simd::V8f;
using simd::V8i;