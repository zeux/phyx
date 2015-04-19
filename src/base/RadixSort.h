#pragma once

inline unsigned int radixUnsignedInt(unsigned int v)
{
	return v;
}

inline unsigned int radixInt(int v)
{
	// flip sign bit
	return static_cast<unsigned int>(v) ^ 0x80000000;
}

inline unsigned int radixUnsignedFloat(const float& v)
{
	return *reinterpret_cast<const unsigned int*>(&v);
}

inline unsigned int radixFloat(const float& v)
{
	// if sign bit is 0, flip sign bit
	// if sign bit is 1, flip everything
	unsigned int f = *reinterpret_cast<const unsigned int*>(&v);
	unsigned int mask = -int(f >> 31) | 0x80000000;
	return f ^ mask;
}

template <typename T, typename Pred> inline T* radixSort3(T* e0, T* e1, size_t count, Pred pred)
{
	unsigned int h[2048*3];

	for (size_t i = 0; i < 2048*3; ++i) h[i] = 0;

	unsigned int* h0 = h;
	unsigned int* h1 = h + 2048;
	unsigned int* h2 = h + 2048*2;

	T* e0_end = e0 + count;

	#define _0(h) ((h) & 2047)
	#define _1(h) (((h) >> 11) & 2047)
	#define _2(h) ((h) >> 22)

	// fill histogram
	for (const T* i = e0; i != e0_end; ++i)
	{
		unsigned int h = pred(*i);

		h0[_0(h)]++; h1[_1(h)]++; h2[_2(h)]++;
	}

	// compute offsets
	{
		unsigned int sum0 = 0, sum1 = 0, sum2 = 0;

		for (unsigned int i = 0; i < 2048; ++i)
		{
			unsigned int c0 = h0[i];
			unsigned int c1 = h1[i];
			unsigned int c2 = h2[i];

			h0[i] = sum0;
			h1[i] = sum1;
			h2[i] = sum2;

			sum0 += c0;
			sum1 += c1;
			sum2 += c2;
		}
	}

	for (size_t i = 0; i < count; ++i)
	{
		unsigned int h = pred(e0[i]);
		e1[h0[_0(h)]++] = e0[i];
	}

	for (size_t i = 0; i < count; ++i)
	{
		unsigned int h = pred(e1[i]);
		e0[h1[_1(h)]++] = e1[i];
	}

	for (size_t i = 0; i < count; ++i)
	{
		unsigned int h = pred(e0[i]);
		e1[h2[_2(h)]++] = e0[i];
	}

	#undef _0
	#undef _1
	#undef _2

	return e1;
}

template <typename T, typename Pred> inline T* radixSort4(T* e0, T* e1, size_t count, Pred pred)
{
	unsigned int h[256*4];

	for (size_t i = 0; i < 256*4; ++i) h[i] = 0;

	unsigned int* h0 = h;
	unsigned int* h1 = h + 256;
	unsigned int* h2 = h + 256*2;
	unsigned int* h3 = h + 256*3;

	T* e0_end = e0 + count;

	#define _0(h) ((h) & 255)
	#define _1(h) (((h) >> 8) & 255)
	#define _2(h) (((h) >> 16) & 255)
	#define _3(h) ((h) >> 24)

	// fill histogram
	for (const T* i = e0; i != e0_end; ++i)
	{
		unsigned int h = pred(*i);

		h0[_0(h)]++; h1[_1(h)]++; h2[_2(h)]++; h3[_3(h)]++;
	}

	// compute offsets
	{
		unsigned int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

		for (unsigned int i = 0; i < 256; ++i)
		{
			unsigned int c0 = h0[i];
			unsigned int c1 = h1[i];
			unsigned int c2 = h2[i];
			unsigned int c3 = h3[i];

			h0[i] = sum0;
			h1[i] = sum1;
			h2[i] = sum2;
			h3[i] = sum3;

			sum0 += c0;
			sum1 += c1;
			sum2 += c2;
			sum3 += c3;
		}
	}

	for (size_t i = 0; i < count; ++i)
	{
		unsigned int h = pred(e0[i]);
		e1[h0[_0(h)]++] = e0[i];
	}

	for (size_t i = 0; i < count; ++i)
	{
		unsigned int h = pred(e1[i]);
		e0[h1[_1(h)]++] = e1[i];
	}

	for (size_t i = 0; i < count; ++i)
	{
		unsigned int h = pred(e0[i]);
		e1[h2[_2(h)]++] = e0[i];
	}

	for (size_t i = 0; i < count; ++i)
	{
		unsigned int h = pred(e1[i]);
		e0[h3[_3(h)]++] = e1[i];
	}

	#undef _0
	#undef _1
	#undef _2
	#undef _3

	return e0;
}