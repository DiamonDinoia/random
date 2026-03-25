#pragma once

#if defined(_MSC_VER)
#  define PRNG_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#  define PRNG_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#  define PRNG_ALWAYS_INLINE inline
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define PRNG_FLATTEN __attribute__((flatten))
#else
#  define PRNG_FLATTEN
#endif

#if defined(_MSC_VER)
#  define PRNG_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#  define PRNG_RESTRICT __restrict__
#else
#  define PRNG_RESTRICT
#endif


#if defined(_MSC_VER)
#  define PRNG_NEVER_INLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#  define PRNG_NEVER_INLINE __attribute__((cold,noinline))
#else
#  define PRNG_NEVER_INLINE
#endif
