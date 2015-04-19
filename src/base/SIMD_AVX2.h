#pragma once

#include "SIMD_AVX2_Transpose.h"

namespace simd
{
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

		SIMD_INLINE static V8f zero()
		{
			return _mm256_setzero_ps();
		}

		SIMD_INLINE static V8f one(float v)
		{
			return _mm256_set1_ps(v);
		}

		SIMD_INLINE static V8f sign()
		{
			return _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
		}

		SIMD_INLINE static V8f load(const float* ptr)
		{
			return _mm256_load_ps(ptr);
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

		SIMD_INLINE static V8i zero()
		{
			return _mm256_setzero_si256();
		}

		SIMD_INLINE static V8i one(int v)
		{
			return _mm256_set1_epi32(v);
		}

		SIMD_INLINE static V8i load(const int* ptr)
		{
			return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
		}
	};

	struct V8b
	{
		__m256 v;

		SIMD_INLINE V8b()
		{
		}

		SIMD_INLINE V8b(__m256 v): v(v)
		{
		}

		SIMD_INLINE V8b(__m256i v): v(_mm256_castsi256_ps(v))
		{
		}

		SIMD_INLINE operator __m256() const
		{
			return v;
		}

		SIMD_INLINE static V8b zero()
		{
			return _mm256_setzero_ps();
		}
	};

	SIMD_INLINE V8i bitcast(V8f v)
	{
		return _mm256_castps_si256(v.v);
	}

	SIMD_INLINE V8f bitcast(V8i v)
	{
		return _mm256_castsi256_ps(v.v);
	}

	SIMD_INLINE V8f operator+(V8f v)
	{
		return v;
	}

	SIMD_INLINE V8f operator-(V8f v)
	{
		return _mm256_xor_ps(V8f::sign(), v.v);
	}

	SIMD_INLINE V8f operator+(V8f l, V8f r)
	{
		return _mm256_add_ps(l.v, r.v);
	}

	SIMD_INLINE V8f operator-(V8f l, V8f r)
	{
		return _mm256_sub_ps(l.v, r.v);
	}

	SIMD_INLINE V8f operator*(V8f l, V8f r)
	{
		return _mm256_mul_ps(l.v, r.v);
	}

	SIMD_INLINE V8f operator/(V8f l, V8f r)
	{
		return _mm256_div_ps(l.v, r.v);
	}

	SIMD_INLINE void operator+=(V8f& l, V8f r)
	{
		l.v = _mm256_add_ps(l.v, r.v);
	}

	SIMD_INLINE void operator-=(V8f& l, V8f r)
	{
		l.v = _mm256_sub_ps(l.v, r.v);
	}

	SIMD_INLINE void operator*=(V8f& l, V8f r)
	{
		l.v = _mm256_mul_ps(l.v, r.v);
	}

	SIMD_INLINE void operator/=(V8f& l, V8f r)
	{
		l.v = _mm256_div_ps(l.v, r.v);
	}

	SIMD_INLINE V8b operator==(V8f l, V8f r)
	{
		return _mm256_cmp_ps(l.v, r.v, _CMP_EQ_UQ);
	}

	SIMD_INLINE V8b operator==(V8i l, V8i r)
	{
		return _mm256_cmpeq_epi32(l.v, r.v);
	}

	SIMD_INLINE V8b operator!=(V8f l, V8f r)
	{
		return _mm256_cmp_ps(l.v, r.v, _CMP_NEQ_UQ);
	}

	SIMD_INLINE V8b operator!=(V8i l, V8i r)
	{
		return _mm256_xor_si256(_mm256_setzero_si256(), _mm256_cmpeq_epi32(l.v, r.v));
	}

	SIMD_INLINE V8b operator<(V8f l, V8f r)
	{
		return _mm256_cmp_ps(l.v, r.v, _CMP_LT_OQ);
	}

	SIMD_INLINE V8b operator<(V8i l, V8i r)
	{
		return _mm256_cmpgt_epi32(r.v, l.v);
	}

	SIMD_INLINE V8b operator<=(V8f l, V8f r)
	{
		return _mm256_cmp_ps(l.v, r.v, _CMP_LE_OQ);
	}

	SIMD_INLINE V8b operator<=(V8i l, V8i r)
	{
		return _mm256_xor_si256(_mm256_setzero_si256(), _mm256_cmpgt_epi32(l.v, r.v));
	}

	SIMD_INLINE V8b operator>(V8f l, V8f r)
	{
		return _mm256_cmp_ps(l.v, r.v, _CMP_GT_OQ);
	}

	SIMD_INLINE V8b operator>(V8i l, V8i r)
	{
		return _mm256_cmpgt_epi32(l.v, r.v);
	}

	SIMD_INLINE V8b operator>=(V8f l, V8f r)
	{
		return _mm256_cmp_ps(l.v, r.v, _CMP_GE_OQ);
	}

	SIMD_INLINE V8b operator>=(V8i l, V8i r)
	{
		return _mm256_xor_si256(_mm256_setzero_si256(), _mm256_cmpgt_epi32(r.v, l.v));
	}

	SIMD_INLINE V8b operator!(V8b v)
	{
		return _mm256_xor_ps(_mm256_setzero_ps(), v.v);
	}

	SIMD_INLINE V8b operator&(V8b l, V8b r)
	{
		return _mm256_and_ps(l.v, r.v);
	}

	SIMD_INLINE V8b operator|(V8b l, V8b r)
	{
		return _mm256_or_ps(l.v, r.v);
	}

	SIMD_INLINE V8b operator^(V8b l, V8b r)
	{
		return _mm256_xor_ps(l.v, r.v);
	}

	SIMD_INLINE void operator&=(V8b& l, V8b r)
	{
		l.v = _mm256_and_ps(l.v, r.v);
	}

	SIMD_INLINE void operator|=(V8b& l, V8b r)
	{
		l.v = _mm256_or_ps(l.v, r.v);
	}

	SIMD_INLINE void operator^=(V8b& l, V8b r)
	{
		l.v = _mm256_xor_ps(l.v, r.v);
	}

	SIMD_INLINE V8f abs(V8f v)
	{
		return _mm256_andnot_ps(V8f::sign(), v.v);
	}

	SIMD_INLINE V8f copysign(V8f x, V8f y)
	{
		V8f sign = V8f::sign();

		return _mm256_or_ps(_mm256_andnot_ps(sign.v, x.v), _mm256_and_ps(y.v, sign.v));
	}

	SIMD_INLINE V8f flipsign(V8f x, V8f y)
	{
		return _mm256_xor_ps(x.v, _mm256_and_ps(y.v, V8f::sign()));
	}

	SIMD_INLINE V8f min(V8f l, V8f r)
	{
		return _mm256_min_ps(l.v, r.v);
	}

	SIMD_INLINE V8f max(V8f l, V8f r)
	{
		return _mm256_max_ps(l.v, r.v);
	}

	SIMD_INLINE V8f select(V8f l, V8f r, V8b m)
	{
		return _mm256_blendv_ps(l.v, r.v, m.v);
	}

	SIMD_INLINE V8i select(V8i l, V8i r, V8b m)
	{
		__m256i mi = _mm256_castps_si256(m.v);

		return _mm256_blendv_epi8(l.v, r.v, mi);
	}

	SIMD_INLINE bool none(V8b v)
	{
		return _mm256_movemask_ps(v.v) == 0;
	}

	SIMD_INLINE bool any(V8b v)
	{
		return _mm256_movemask_ps(v.v) != 0;
	}

	SIMD_INLINE bool all(V8b v)
	{
		return _mm256_movemask_ps(v.v) == 31;
	}

	SIMD_INLINE void store(V8f v, float* ptr)
	{
		_mm256_store_ps(ptr, v.v);
	}

	SIMD_INLINE void store(V8i v, int* ptr)
	{
		_mm256_store_si256(reinterpret_cast<__m256i*>(ptr), v.v);
	}

	SIMD_INLINE void loadindexed4(V8f& v0, V8f& v1, V8f& v2, V8f& v3, const void* base, const int indices[8], unsigned int stride)
		{
		const char* ptr = static_cast<const char*>(base);

		__m128 hr0 = _mm_load_ps(reinterpret_cast<const float*>(ptr + indices[0] * stride));
		__m128 hr1 = _mm_load_ps(reinterpret_cast<const float*>(ptr + indices[1] * stride));
		__m128 hr2 = _mm_load_ps(reinterpret_cast<const float*>(ptr + indices[2] * stride));
		__m128 hr3 = _mm_load_ps(reinterpret_cast<const float*>(ptr + indices[3] * stride));
		__m128 hr4 = _mm_load_ps(reinterpret_cast<const float*>(ptr + indices[4] * stride));
		__m128 hr5 = _mm_load_ps(reinterpret_cast<const float*>(ptr + indices[5] * stride));
		__m128 hr6 = _mm_load_ps(reinterpret_cast<const float*>(ptr + indices[6] * stride));
		__m128 hr7 = _mm_load_ps(reinterpret_cast<const float*>(ptr + indices[7] * stride));

		__m256 r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(hr0), hr4, 1);
		__m256 r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(hr1), hr5, 1);
		__m256 r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(hr2), hr6, 1);
		__m256 r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(hr3), hr7, 1);

		_MM_TRANSPOSE8_LANE4_PS(r0, r1, r2, r3);

		v0.v = r0;
		v1.v = r1;
		v2.v = r2;
		v3.v = r3;
	}

	SIMD_INLINE void storeindexed4(const V8f& v0, const V8f& v1, const V8f& v2, const V8f& v3, void* base, const int indices[8], unsigned int stride)
	{
		char* ptr = static_cast<char*>(base);

		__m256 r0 = v0.v;
		__m256 r1 = v1.v;
		__m256 r2 = v2.v;
		__m256 r3 = v3.v;

		_MM_TRANSPOSE8_LANE4_PS(r0, r1, r2, r3);

		__m128 hr0 = _mm256_castps256_ps128(r0);
		__m128 hr1 = _mm256_castps256_ps128(r1);
		__m128 hr2 = _mm256_castps256_ps128(r2);
		__m128 hr3 = _mm256_castps256_ps128(r3);
		__m128 hr4 = _mm256_extractf128_ps(r0, 1);
		__m128 hr5 = _mm256_extractf128_ps(r1, 1);
		__m128 hr6 = _mm256_extractf128_ps(r2, 1);
		__m128 hr7 = _mm256_extractf128_ps(r3, 1);

		_mm_store_ps(reinterpret_cast<float*>(ptr + indices[0] * stride), hr0);
		_mm_store_ps(reinterpret_cast<float*>(ptr + indices[1] * stride), hr1);
		_mm_store_ps(reinterpret_cast<float*>(ptr + indices[2] * stride), hr2);
		_mm_store_ps(reinterpret_cast<float*>(ptr + indices[3] * stride), hr3);
		_mm_store_ps(reinterpret_cast<float*>(ptr + indices[4] * stride), hr4);
		_mm_store_ps(reinterpret_cast<float*>(ptr + indices[5] * stride), hr5);
		_mm_store_ps(reinterpret_cast<float*>(ptr + indices[6] * stride), hr6);
		_mm_store_ps(reinterpret_cast<float*>(ptr + indices[7] * stride), hr7);
	}
}

namespace simd
{
	template <> struct VNf_<8> { typedef V8f type; };
	template <> struct VNi_<8> { typedef V8i type; };
	template <> struct VNb_<8> { typedef V8b type; };
}

using simd::V8f;
using simd::V8i;
using simd::V8b;