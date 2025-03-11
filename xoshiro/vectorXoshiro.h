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

  constexpr simd_type next() noexcept {
    const simd_type result = rotl(m_state[0] + m_state[3], 23) + m_state[0];
    //
    const simd_type t = m_state[1] << 17;

    m_state[2] ^= m_state[0];
    m_state[3] ^= m_state[1];
    m_state[1] ^= m_state[2];
    m_state[0] ^= m_state[3];

    m_state[2] ^= t;

    m_state[3] = rotl(m_state[3], 45);

    return result;
  }

  constexpr void populate_cache() noexcept {
    for (auto i = 0UL; i < 1024; i+=SIMD_WIDTH) {
      next().store_aligned(m_cache.data() + i);
    }
    m_index = 0;
  }

public:
  /* This is the jump function for the generator. It is equivalent
 to 2^128 calls to next(); it can be used to generate 2^128
 non-overlapping subsequences for parallel computations. */
  constexpr void jump() noexcept {
    constexpr std::uint64_t JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
                                      0xa9582618e03fc9aa, 0x39abdc4529b1661c};

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