#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include <xsimd/xsimd.hpp>

#include "xoshiro.hpp"

#include <iostream>

namespace xoshiro {

class VectorXoshiro {
  using result_type = std::uint64_t;
  using simd_type = xsimd::simd_type<result_type>;
  static constexpr auto RNG_WIDTH = 4U;

public:
  static constexpr auto SIMD_WIDTH = xsimd::simd_type<result_type>::size;
  constexpr explicit VectorXoshiro(const std::uint64_t seed) noexcept
      : m_state{}, m_cache{}, m_index(1024) {
    Xoshiro rng{seed};
    std::array<std::array<std::uint64_t, SIMD_WIDTH>, RNG_WIDTH> states{};
    for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
      for (auto j = 0UL; j < RNG_WIDTH; ++j) {
        states[j][i] = rng.getState()[j];
      }
      rng.jump();
    }
    for (auto i = 0UL; i < RNG_WIDTH; ++i) {
      m_state[i] = simd_type::load_unaligned(states[i].data());
    }
  }

  constexpr explicit VectorXoshiro(const std::uint64_t seed,
                                   const std::uint64_t thread_id) noexcept
      : VectorXoshiro(seed) {
    for (auto i = UINT64_C(0); i < thread_id; ++i) {
      jump();
    }
  }

  constexpr std::uint64_t operator()() noexcept {
    if (m_index == 1024) {
      populate_cache();
    }
    return m_cache[m_index++];
  }

  constexpr double uniform() noexcept {
    return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
  }

  constexpr auto getState(const std::size_t index) const {
    std::array<std::uint64_t, RNG_WIDTH> state{};
    for (auto i = 0UL; i < RNG_WIDTH; ++i) {
      state[i] = m_state[i].get(index);
    }
    return state;
  }

  static constexpr result_type(min)() noexcept {
    return (std::numeric_limits<result_type>::min)();
  }

  static constexpr result_type(max)() noexcept {
    return (std::numeric_limits<result_type>::max)();
  }

  static constexpr std::uint64_t stateSize() noexcept { return RNG_WIDTH; }

private:
  std::array<simd_type, RNG_WIDTH> m_state;

  std::array<std::uint64_t, 1024> m_cache;
  std::size_t m_index;

  // Define a SIMD type alias based on available extensions.
#ifdef __AVX512F__
  using simd_u64 = __m512i;
#elif defined(__AVX2__)
  using simd_u64 = __m256i;
#elif defined(__SSE2__)
  using simd_u64 = __m128i;
#else
#error "No supported SIMD extensions available."
#endif

  // Constant rotate left for 64-bit lanes.
  // Shift must be a compile-time constant in [0, 63].
  template <int Shift>
  inline simd_u64 rotl_u64(const simd_u64 x) {
    static_assert(Shift >= 0 && Shift < 64, "Shift must be in [0, 63]");
#if defined(__AVX512F__)
#if defined(__AVX512BITALG__)
    // Use the dedicated AVX-512 BITALG rotate instruction.
    return _mm512_rol_epi64(x, Shift);
#else
    // Fallback: combine shifts and OR.
    return _mm512_or_si512(_mm512_slli_epi64(x, Shift),
                           _mm512_srli_epi64(x, 64 - Shift));
#endif
#elif defined(__AVX2__)
    return _mm256_or_si256(_mm256_slli_epi64(x, Shift),
                           _mm256_srli_epi64(x, 64 - Shift));
#elif defined(__SSE2__)
    return _mm_or_si128(_mm_slli_epi64(x, Shift),
                        _mm_srli_epi64(x, 64 - Shift));
#endif
  }

  constexpr auto next() noexcept {
    const auto result = rotl_u64<23>(m_state[0] + m_state[3]) + m_state[0];
    const auto t = m_state[1] << 17;

    m_state[2] ^= m_state[0];
    m_state[3] ^= m_state[1];
    m_state[1] ^= m_state[2];
    m_state[0] ^= m_state[3];

    m_state[2] ^= t;

    m_state[3] = rotl_u64<45>(m_state[3]);

    return result;
  }

public:
  constexpr void populate_cache() noexcept {
    for (auto i = 0UL; i < 1024UL; i+=SIMD_WIDTH) {
      next().store_aligned(m_cache.data() + i);
    }
    m_index = 0;
  }

  /* This is the jump function for the generator. It is equivalent
 to 2^128 calls to next(); it can be used to generate 2^128
 non-overlapping subsequences for parallel computations. */
  constexpr void jump() noexcept {
    constexpr std::uint64_t JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
                                      0xa9582618e03fc9aa, 0x39abdc4529b1661c};
    for (auto i = 0; i < SIMD_WIDTH; ++i) {
      simd_type s0(0);
      simd_type s1(0);
      simd_type s2(0);
      simd_type s3(0);
      for (const auto i : JUMP)
        for (auto b = 0; b < 64; b++) {
          if (i & std::uint64_t{1} << b) {
            s0 ^= m_state[0];
            s1 ^= m_state[1];
            s2 ^= m_state[2];
            s3 ^= m_state[3];
          }
          next();
        }
      m_state[0] = s0;
      m_state[1] = s1;
      m_state[2] = s2;
      m_state[3] = s3;
    }
  }

  /* This is the long-jump function for the generator. It is equivalent to
 2^192 calls to next(); it can be used to generate 2^64 starting points,
 from each of which jump() will generate 2^64 non-overlapping
 subsequences for parallel distributed computations. */
  constexpr void long_jump() noexcept {
    constexpr std::uint64_t LONG_JUMP[] = {
        0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241,
        0x39109bb02acbe635};
    simd_type s0(0);
    simd_type s1(0);
    simd_type s2(0);
    simd_type s3(0);
    for (const auto i : LONG_JUMP)
      for (auto b = 0; b < 64; b++) {
        if (i & std::uint64_t{1} << b) {
          s0 ^= m_state[0];
          s1 ^= m_state[1];
          s2 ^= m_state[2];
          s3 ^= m_state[3];
        }
        next();
      }
    m_state[0] = s0;
    m_state[1] = s1;
    m_state[2] = s2;
    m_state[3] = s3;
  }
};

} // namespace xoshiro