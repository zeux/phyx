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

		SIMD_INLINE static V4f zero()
		{
			return _mm_setzero_ps();
		}

		SIMD_INLINE static V4f one(float v)
		{
			return _mm_set1_ps(v);
		}

		SIMD_INLINE static V4f sign()
		{
			return _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
		}

		SIMD_INLINE static V4f load(const float* ptr)
		{
			return _mm_load_ps(ptr);
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

		SIMD_INLINE static V4i zero()
		{
			return _mm_setzero_si128();
		}

		SIMD_INLINE static V4i one(int v)
		{
			return _mm_set1_epi32(v);
		}

		SIMD_INLINE static V4i load(const int* ptr)
		{
			return _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
		}
	};

	struct V4b
	{
		__m128 v;

		SIMD_INLINE V4b()
		{
		}

		SIMD_INLINE V4b(__m128 v): v(v)
		{
		}

		SIMD_INLINE V4b(__m128i v): v(_mm_castsi128_ps(v))
		{
		}

		SIMD_INLINE operator __m128() const
		{
			return v;
		}

		SIMD_INLINE static V4b zero()
		{
			return _mm_setzero_ps();
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

	SIMD_INLINE V4f operator+(V4f v)
	{
		return v;
	}

	SIMD_INLINE V4f operator-(V4f v)
	{
		return _mm_xor_ps(V4f::sign(), v.v);
	}

	SIMD_INLINE V4f operator+(V4f l, V4f r)
	{
		return _mm_add_ps(l.v, r.v);
	}

	SIMD_INLINE V4f operator-(V4f l, V4f r)
	{
		return _mm_sub_ps(l.v, r.v);
	}

	SIMD_INLINE V4f operator*(V4f l, V4f r)
	{
		return _mm_mul_ps(l.v, r.v);
	}

	SIMD_INLINE V4f operator/(V4f l, V4f r)
	{
		return _mm_div_ps(l.v, r.v);
	}

	SIMD_INLINE void operator+=(V4f& l, V4f r)
	{
		l.v = _mm_add_ps(l.v, r.v);
	}

	SIMD_INLINE void operator-=(V4f& l, V4f r)
	{
		l.v = _mm_sub_ps(l.v, r.v);
	}

	SIMD_INLINE void operator*=(V4f& l, V4f r)
	{
		l.v = _mm_mul_ps(l.v, r.v);
	}

	SIMD_INLINE void operator/=(V4f& l, V4f r)
	{
		l.v = _mm_div_ps(l.v, r.v);
	}

	SIMD_INLINE V4b operator==(V4f l, V4f r)
	{
		return _mm_cmpeq_ps(l.v, r.v);
	}

	SIMD_INLINE V4b operator==(V4i l, V4i r)
	{
		return _mm_cmpeq_epi32(l.v, r.v);
	}

	SIMD_INLINE V4b operator!=(V4f l, V4f r)
	{
		return _mm_cmpneq_ps(l.v, r.v);
	}

	SIMD_INLINE V4b operator!=(V4i l, V4f r)
	{
		return _mm_xor_si128(_mm_setzero_si128(), _mm_cmpeq_epi32(l.v, r.v));
	}

	SIMD_INLINE V4b operator<(V4f l, V4f r)
	{
		return _mm_cmplt_ps(l.v, r.v);
	}

	SIMD_INLINE V4b operator<(V4i l, V4i r)
	{
		return _mm_cmplt_epi32(l.v, r.v);
	}

	SIMD_INLINE V4b operator<=(V4f l, V4f r)
	{
		return _mm_cmple_ps(l.v, r.v);
	}

	SIMD_INLINE V4b operator<=(V4i l, V4i r)
	{
		return _mm_xor_si128(_mm_setzero_si128(), _mm_cmplt_epi32(r.v, l.v));
	}

	SIMD_INLINE V4b operator>(V4f l, V4f r)
	{
		return _mm_cmpgt_ps(l.v, r.v);
	}

	SIMD_INLINE V4b operator>(V4i l, V4i r)
	{
		return _mm_cmpgt_epi32(l.v, r.v);
	}

	SIMD_INLINE V4b operator>=(V4f l, V4f r)
	{
		return _mm_cmpge_ps(l.v, r.v);
	}

	SIMD_INLINE V4b operator>=(V4i l, V4i r)
	{
		return _mm_xor_si128(_mm_setzero_si128(), _mm_cmplt_epi32(l.v, r.v));
	}

	SIMD_INLINE V4b operator!(V4b v)
	{
		return _mm_xor_ps(_mm_setzero_ps(), v.v);
	}

	SIMD_INLINE V4b operator&(V4b l, V4b r)
	{
		return _mm_and_ps(l.v, r.v);
	}

	SIMD_INLINE V4b operator|(V4b l, V4b r)
	{
		return _mm_or_ps(l.v, r.v);
	}

	SIMD_INLINE V4b operator^(V4b l, V4b r)
	{
		return _mm_xor_ps(l.v, r.v);
	}

	SIMD_INLINE void operator&=(V4b& l, V4b r)
	{
		l.v = _mm_and_ps(l.v, r.v);
	}

	SIMD_INLINE void operator|=(V4b& l, V4b r)
	{
		l.v = _mm_or_ps(l.v, r.v);
	}

	SIMD_INLINE void operator^=(V4b l, V4b r)
	{
		l.v = _mm_xor_ps(l.v, r.v);
	}

	SIMD_INLINE V4f abs(V4f v)
	{
		return _mm_andnot_ps(V4f::sign(), v.v);
	}

	SIMD_INLINE V4f copysign(V4f x, V4f y)
	{
		V4f sign = V4f::sign();

		return _mm_or_ps(_mm_andnot_ps(sign.v, x.v), _mm_and_ps(y.v, sign.v));
	}

	SIMD_INLINE V4f flipsign(V4f x, V4f y)
	{
		return _mm_xor_ps(x.v, _mm_and_ps(y.v, V4f::sign()));
	}

	SIMD_INLINE V4f min(V4f l, V4f r)
	{
		return _mm_min_ps(l.v, r.v);
	}

	SIMD_INLINE V4f max(V4f l, V4f r)
	{
		return _mm_max_ps(l.v, r.v);
	}

	SIMD_INLINE V4f select(V4f l, V4f r, V4b m)
	{
		return _mm_or_ps(_mm_andnot_ps(m.v, l.v), _mm_and_ps(r.v, m.v));
	}

	SIMD_INLINE V4i select(V4i l, V4i r, V4b m)
	{
		__m128i mi = _mm_castps_si128(m.v);

		return _mm_or_si128(_mm_andnot_si128(mi, l.v), _mm_and_si128(r.v, mi));
	}

	SIMD_INLINE bool none(V4b v)
	{
		return _mm_movemask_ps(v.v) == 0;
	}

	SIMD_INLINE bool any(V4b v)
	{
		return _mm_movemask_ps(v.v) != 0;
	}

	SIMD_INLINE bool all(V4b v)
	{
		return _mm_movemask_ps(v.v) == 15;
	}

	SIMD_INLINE void store(V4f v, float* ptr)
	{
		_mm_store_ps(ptr, v.v);
	}

	SIMD_INLINE void store(V4i v, int* ptr)
	{
		_mm_store_si128(reinterpret_cast<__m128i*>(ptr), v.v);
	}

	SIMD_INLINE void loadindexed4(V4f& v0, V4f& v1, V4f& v2, V4f& v3, const void* base, const int indices[4], unsigned int stride)
	{
		const char* ptr = static_cast<const char*>(base);

		__m128 r0 = _mm_load_ps(reinterpret_cast<const float*>(ptr + indices[0] * stride));
		__m128 r1 = _mm_load_ps(reinterpret_cast<const float*>(ptr + indices[1] * stride));
		__m128 r2 = _mm_load_ps(reinterpret_cast<const float*>(ptr + indices[2] * stride));
		__m128 r3 = _mm_load_ps(reinterpret_cast<const float*>(ptr + indices[3] * stride));

		_MM_TRANSPOSE4_PS(r0, r1, r2, r3);

		v0.v = r0;
		v1.v = r1;
		v2.v = r2;
		v3.v = r3;
	}

	SIMD_INLINE void storeindexed4(const V4f& v0, const V4f& v1, const V4f& v2, const V4f& v3, void* base, const int indices[4], unsigned int stride)
	{
		char* ptr = static_cast<char*>(base);

		__m128 r0 = v0.v;
		__m128 r1 = v1.v;
		__m128 r2 = v2.v;
		__m128 r3 = v3.v;

		_MM_TRANSPOSE4_PS(r0, r1, r2, r3);

		_mm_store_ps(reinterpret_cast<float*>(ptr + indices[0] * stride), r0);
		_mm_store_ps(reinterpret_cast<float*>(ptr + indices[1] * stride), r1);
		_mm_store_ps(reinterpret_cast<float*>(ptr + indices[2] * stride), r2);
		_mm_store_ps(reinterpret_cast<float*>(ptr + indices[3] * stride), r3);
	}
}

using simd::V4f;
using simd::V4i;
using simd::V4b;