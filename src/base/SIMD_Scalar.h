#pragma once

#include <math.h>

namespace simd
{
	struct V1f
	{
		float v;

		SIMD_INLINE V1f()
		{
		}

		SIMD_INLINE V1f(float v): v(v)
		{
		}

		SIMD_INLINE operator float() const
		{
			return v;
		}

		SIMD_INLINE static V1f zero()
		{
			return 0.f;
		}

		SIMD_INLINE static V1f one(float v)
		{
			return v;
		}

		SIMD_INLINE static V1f load(const float* ptr)
		{
			return *ptr;
		}
	};

	struct V1i
	{
		int v;

		SIMD_INLINE V1i()
		{
		}

		SIMD_INLINE V1i(int v): v(v)
		{
		}

		SIMD_INLINE operator int() const
		{
			return v;
		}

		SIMD_INLINE static V1i zero()
		{
			return 0;
		}

		SIMD_INLINE static V1i one(int v)
		{
			return v;
		}

		SIMD_INLINE static V1i load(const int* ptr)
		{
			return *ptr;
		}
	};

	struct V1b
	{
		bool v;

		SIMD_INLINE V1b()
		{
		}

		SIMD_INLINE V1b(bool v): v(v)
		{
		}

		SIMD_INLINE operator bool() const
		{
			return v;
		}

		SIMD_INLINE static V1b zero()
		{
			return false;
		}
	};

	SIMD_INLINE V1f bitcast(const V1i& v)
	{
		return *reinterpret_cast<const float*>(&v.v);
	}

	SIMD_INLINE V1i bitcast(const V1f& v)
	{
		return *reinterpret_cast<const int*>(&v.v);
	}

	SIMD_INLINE V1f operator+(V1f v)
	{
		return v;
	}

	SIMD_INLINE V1f operator-(V1f v)
	{
		return -v.v;
	}

	SIMD_INLINE V1f operator+(V1f l, V1f r)
	{
		return l.v + r.v;
	}

	SIMD_INLINE V1f operator-(V1f l, V1f r)
	{
		return l.v - r.v;
	}

	SIMD_INLINE V1f operator*(V1f l, V1f r)
	{
		return l.v * r.v;
	}

	SIMD_INLINE V1f operator/(V1f l, V1f r)
	{
		return l.v / r.v;
	}

	SIMD_INLINE void operator+=(V1f& l, V1f r)
	{
		l.v += r.v;
	}

	SIMD_INLINE void operator-=(V1f& l, V1f r)
	{
		l.v -= r.v;
	}

	SIMD_INLINE void operator*=(V1f& l, V1f r)
	{
		l.v *= r.v;
	}

	SIMD_INLINE void operator/=(V1f& l, V1f r)
	{
		l.v /= r.v;
	}

	SIMD_INLINE V1b operator==(V1f l, V1f r)
	{
		return l.v == r.v;
	}

	SIMD_INLINE V1b operator==(V1i l, V1i r)
	{
		return l.v == r.v;
	}

	SIMD_INLINE V1b operator!=(V1f l, V1f r)
	{
		return l.v != r.v;
	}

	SIMD_INLINE V1b operator!=(V1i l, V1f r)
	{
		return l.v != r.v;
	}

	SIMD_INLINE V1b operator<(V1f l, V1f r)
	{
		return l.v < r.v;
	}

	SIMD_INLINE V1b operator<(V1i l, V1i r)
	{
		return l.v < r.v;
	}

	SIMD_INLINE V1b operator<=(V1f l, V1f r)
	{
		return l.v <= r.v;
	}

	SIMD_INLINE V1b operator<=(V1i l, V1i r)
	{
		return l.v <= r.v;
	}

	SIMD_INLINE V1b operator>(V1f l, V1f r)
	{
		return l.v > r.v;
	}

	SIMD_INLINE V1b operator>(V1i l, V1i r)
	{
		return l.v > r.v;
	}

	SIMD_INLINE V1b operator>=(V1f l, V1f r)
	{
		return l.v >= r.v;
	}

	SIMD_INLINE V1b operator>=(V1i l, V1i r)
	{
		return l.v >= r.v;
	}

	SIMD_INLINE V1b operator!(V1b v)
	{
		return !v.v;
	}

	SIMD_INLINE V1b operator&(V1b l, V1b r)
	{
		return l.v & r.v;
	}

	SIMD_INLINE V1b operator|(V1b l, V1b r)
	{
		return l.v | r.v;
	}

	SIMD_INLINE V1b operator^(V1b l, V1b r)
	{
		return l.v ^ r.v;
	}

	SIMD_INLINE void operator&=(V1b& l, V1b r)
	{
		l.v &= r.v;
	}

	SIMD_INLINE void operator|=(V1b& l, V1b r)
	{
		l.v |= r.v;
	}

	SIMD_INLINE void operator^=(V1b l, V1b r)
	{
		l.v ^= r.v;
	}

	SIMD_INLINE V1f abs(V1f v)
	{
		return fabsf(v.v);
	}

	SIMD_INLINE V1f copysign(V1f x, V1f y)
	{
		return copysignf(x, y);
	}

	SIMD_INLINE V1f flipsign(V1f x, V1f y)
	{
		return y.v < 0.f ? -x.v : x.v;
	}

	SIMD_INLINE V1f min(V1f l, V1f r)
	{
		return l.v < r.v ? l.v : r.v;
	}

	SIMD_INLINE V1f max(V1f l, V1f r)
	{
		return l.v > r.v ? l.v : r.v;
	}

	SIMD_INLINE V1f select(V1f l, V1f r, V1b m)
	{
		return m.v ? r.v : l.v;
	}

	SIMD_INLINE V1i select(V1i l, V1i r, V1b m)
	{
		return m.v ? r.v : l.v;
	}

	SIMD_INLINE bool none(V1b v)
	{
		return !v.v;
	}

	SIMD_INLINE bool any(V1b v)
	{
		return v.v;
	}

	SIMD_INLINE bool all(V1b v)
	{
		return v.v;
	}

	SIMD_INLINE void store(V1f v, float* ptr)
	{
		*ptr = v.v;
	}

	SIMD_INLINE void store(V1i v, int* ptr)
	{
		*ptr = v.v;
	}

	SIMD_INLINE void loadindexed4(V1f& v0, V1f& v1, V1f& v2, V1f& v3, const void* base, const int indices[1], unsigned int stride)
	{
		const float* ptr = reinterpret_cast<const float*>(static_cast<const char*>(base) + indices[0] * stride);

		v0.v = ptr[0];
		v1.v = ptr[1];
		v2.v = ptr[2];
		v3.v = ptr[3];
	}

	SIMD_INLINE void storeindexed4(const V1f& v0, const V1f& v1, const V1f& v2, const V1f& v3, void* base, const int indices[1], unsigned int stride)
	{
		float* ptr = reinterpret_cast<float*>(static_cast<char*>(base) + indices[0] * stride);

		ptr[0] = v0.v;
		ptr[1] = v1.v;
		ptr[2] = v2.v;
		ptr[3] = v3.v;
	}

	SIMD_INLINE void loadindexed8(V1f& v0, V1f& v1, V1f& v2, V1f& v3, V1f& v4, V1f& v5, V1f& v6, V1f& v7, const void* base, const int indices[1], unsigned int stride)
	{
		const float* ptr = reinterpret_cast<const float*>(static_cast<const char*>(base) + indices[0] * stride);

		v0.v = ptr[0];
		v1.v = ptr[1];
		v2.v = ptr[2];
		v3.v = ptr[3];
		v4.v = ptr[4];
		v5.v = ptr[5];
		v6.v = ptr[6];
		v7.v = ptr[7];
	}
}

namespace simd
{
	template <> struct VNf_<1> { typedef V1f type; };
	template <> struct VNi_<1> { typedef V1i type; };
	template <> struct VNb_<1> { typedef V1b type; };
}

using simd::V1f;
using simd::V1i;
using simd::V1b;