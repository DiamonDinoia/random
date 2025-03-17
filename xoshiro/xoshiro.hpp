#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include "splitMix64.hpp"

namespace xoshiro {

class Xoshiro {
public:
  constexpr explicit Xoshiro(const std::uint64_t seed) noexcept : m_state{} {
    SplitMix64 splitMix64{seed};
    for (auto &element : m_state) {
      element = splitMix64();
    }
  }

  constexpr explicit Xoshiro(const std::uint64_t seed,
                             const std::uint64_t thread_id) noexcept
      : Xoshiro(seed) {
    for (auto i = UINT64_C(0); i < thread_id; ++i) {
      jump();
    }
  }

  std::uint64_t operator()() noexcept { return next(); }

  double uniform() noexcept {
    return static_cast<double>(next() >> 11) * 0x1.0p-53;
  }

  constexpr std::array<std::uint64_t, 4> getState() const {
    return m_state;
  }

  constexpr void setState(std::array<std::uint64_t, 4>&& state) noexcept {
    m_state = state;
  }

  static constexpr std::uint64_t min() noexcept {
    return std::numeric_limits<std::uint64_t>::min();
  }
  static constexpr std::uint64_t max() noexcept {
    return std::numeric_limits<std::uint64_t>::max();
  }
  static constexpr std::uint64_t stateSize() noexcept { return 4; }

private:
    std::array<std::uint64_t, 4> m_state;

  static constexpr std::uint64_t rotl(const std::uint64_t x, int k) noexcept {
    return (x << k) | (x >> (64 - k));
  }

  std::uint64_t next() noexcept {
    // Compute the shift constant and the output result.
    const auto t_shift = m_state[1] << 17;
    const auto result = rotl(m_state[0] + m_state[3], 23) + m_state[0];

    m_state[2] ^= m_state[0];
    m_state[3] ^= m_state[1];
    m_state[0] ^= m_state[3];
    m_state[1] ^= m_state[2];

    m_state[2] ^= t_shift;
    m_state[3] = rotl(m_state[3], 45);

    return result;
  }


public:
  /* This is the jump function for the generator. It is equivalent
 to 2^128 calls to next(); it can be used to generate 2^128
 non-overlapping subsequences for parallel computations. */
  constexpr void jump() noexcept {
    constexpr std::uint64_t JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
                                      0xa9582618e03fc9aa, 0x39abdc4529b1661c};
    std::uint64_t s0 = 0;
    std::uint64_t s1 = 0;
    std::uint64_t s2 = 0;
    std::uint64_t s3 = 0;
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
    std::uint64_t s0 = 0;
    std::uint64_t s1 = 0;
    std::uint64_t s2 = 0;
    std::uint64_t s3 = 0;
    for (unsigned long i : LONG_JUMP)
      for (int b = 0; b < 64; b++) {
        if (i & UINT64_C(1) << b) {
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

}