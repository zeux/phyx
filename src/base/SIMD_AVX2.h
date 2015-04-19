#pragma once

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

		static V8b zero()
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
}

using simd::V8f;
using simd::V8i;
using simd::V8b;