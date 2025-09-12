#pragma once

#if defined(_MSC_VER)
#  define PRNG_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#  define PRNG_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#  define PRNG_ALWAYS_INLINE inline
#endif


#if defined(_MSC_VER)
#  define PRNG_NEVER_INLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#  define PRNG_NEVER_INLINE __attribute__((cold,noinline))
#else
#  define PRNG_NEVER_INLINE
#endif

