#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <xsimd/xsimd.hpp>

#include "random/macros.hpp"

namespace prng {
template <std::uint8_t R, class Arch>
class ChaChaSIMD {
protected:
  static constexpr auto MATRIX_ROW_LEN = std::uint8_t{4};
  static constexpr auto MATRIX_COL_LEN = std::uint8_t{4};
  static constexpr auto MATRIX_WORDCOUNT = std::uint8_t{16};
  static constexpr auto KEY_WORDCOUNT = std::uint8_t{8};
  static constexpr auto KEY_HALF_WORDCOUNT = std::uint8_t{4};

public:
  using input_word = std::uint64_t;
  using matrix_word = std::uint32_t;
  using matrix_type = std::array<matrix_word, MATRIX_WORDCOUNT>;
  using rounds_type = std::uint8_t;
  using simd_type = xsimd::batch<matrix_word, Arch>;

protected:
  static constexpr std::uint8_t SIMD_WIDTH = std::uint8_t{simd_type::size};
  static constexpr auto CACHE_SIZE = std::uint8_t{SIMD_WIDTH};

public:
  explicit PRNG_ALWAYS_INLINE ChaChaSIMD(
    const std::array<matrix_word, KEY_WORDCOUNT> key,
    const input_word counter,
    const input_word nonce
  ) {
    // TODO: Maybe consider making this constructor shared between both scalar
    // and simd impl as it is the same between both

    // First four words (i.e. top-row) are always the same constants
    // They spell out "expand 2-byte k" in ASCII (little-endian)
    m_state[0] = 0x61707865;
    m_state[1] = 0x3320646e;
    m_state[2] = 0x79622d32;
    m_state[3] = 0x6b206574;

    for (auto i = 0; i < 8; ++i) {
      m_state[4 + i] = key[i];
    }

    // ChaCha assumes little-endianness
    m_state[12] = static_cast<matrix_word>(counter & 0xFFFFFFFF);
    m_state[13] = static_cast<matrix_word>(counter >> 32);
    m_state[14] = static_cast<matrix_word>(nonce & 0xFFFFFFFF);
    m_state[15] = static_cast<matrix_word>(nonce >> 32);
  }

  PRNG_ALWAYS_INLINE constexpr matrix_type operator()() noexcept {
    if (m_index >= CACHE_SIZE) [[unlikely]] {
      gen_next_blocks_in_cache();
      m_index = 0;
    }
    return m_cache[m_index++];
  }


private:
  matrix_type m_state;
  matrix_type m_cache[CACHE_SIZE];
  // Initalize to "past end of the cache" since cache starts empty
  std::uint8_t m_index = CACHE_SIZE;

  // Return an array that's like { 0, 1, ..., N - 1 }. Can be used to initalize a batch
  // for consecutively incremeting a batch of low counter words.
  template <size_t... Is>
  static constexpr PRNG_ALWAYS_INLINE std::array<matrix_word, sizeof...(Is)> make_lower_counter_inc(std::index_sequence<Is...>) noexcept {
    return {Is...};
  }

  PRNG_ALWAYS_INLINE constexpr void gen_next_blocks_in_cache() noexcept {
    simd_type lower_counter_inc = 
      xsimd::load_unaligned(make_lower_counter_inc(std::make_index_sequence<SIMD_WIDTH>{}).data());
    simd_type higher_counter_inc;
    matrix_word c = std::numeric_limits<matrix_word>::max() - m_state[12];
    if (c < SIMD_WIDTH) [[unlikely]] {
      matrix_word b[SIMD_WIDTH];
      b[0] = 0;
      for (auto i = 1; i < SIMD_WIDTH; ++i) {
        b[i] = c < i;
      }
      higher_counter_inc = xsimd::load_unaligned(b);
    } else {
      higher_counter_inc = simd_type::broadcast(0);
    }

    simd_type x[MATRIX_WORDCOUNT];
    for (auto i = 0; i < MATRIX_WORDCOUNT; ++i) {
      x[i] = simd_type::broadcast(m_state[i]);
    }
    x[12] += lower_counter_inc;
    x[13] += higher_counter_inc;

    for (auto i = 0; i < R; i += 2) {
      x[0] += x[4];
      x[1] += x[5];
      x[2] += x[6];
      x[3] += x[7];

      x[12] ^= x[0];
      x[13] ^= x[1];
      x[14] ^= x[2];
      x[15] ^= x[3];

      x[12] = xsimd::rotl(x[12], 16);
      x[13] = xsimd::rotl(x[13], 16);
      x[14] = xsimd::rotl(x[14], 16);
      x[15] = xsimd::rotl(x[15], 16);

      x[8] += x[12];
      x[9] += x[13];
      x[10] += x[14];
      x[11] += x[15];

      x[4] ^= x[8];
      x[5] ^= x[9];
      x[6] ^= x[10];
      x[7] ^= x[11];

      x[4] = xsimd::rotl(x[4], 12);
      x[5] = xsimd::rotl(x[5], 12);
      x[6] = xsimd::rotl(x[6], 12);
      x[7] = xsimd::rotl(x[7], 12);

      x[0] += x[4];
      x[1] += x[5];
      x[2] += x[6];
      x[3] += x[7];

      x[12] ^= x[0];
      x[13] ^= x[1];
      x[14] ^= x[2];
      x[15] ^= x[3];

      x[12] = xsimd::rotl(x[12], 8);
      x[13] = xsimd::rotl(x[13], 8);
      x[14] = xsimd::rotl(x[14], 8);
      x[15] = xsimd::rotl(x[15], 8);

      x[8] += x[12];
      x[9] += x[13];
      x[10] += x[14];
      x[11] += x[15];

      x[4] ^= x[8];
      x[5] ^= x[9];
      x[6] ^= x[10];
      x[7] ^= x[11];

      x[4] = xsimd::rotl(x[4], 7);
      x[5] = xsimd::rotl(x[5], 7);
      x[6] = xsimd::rotl(x[6], 7);
      x[7] = xsimd::rotl(x[7], 7);

      x[0] += x[5];
      x[1] += x[6];
      x[2] += x[7];
      x[3] += x[4];

      x[15] ^= x[0];
      x[12] ^= x[1];
      x[13] ^= x[2];
      x[14] ^= x[3];

      x[15] = xsimd::rotl(x[15], 16);
      x[12] = xsimd::rotl(x[12], 16);
      x[13] = xsimd::rotl(x[13], 16);
      x[14] = xsimd::rotl(x[14], 16);

      x[10] += x[15];
      x[11] += x[12];
      x[8] += x[13];
      x[9] += x[14];

      x[5] ^= x[10];
      x[6] ^= x[11];
      x[7] ^= x[8];
      x[4] ^= x[9];

      x[5] = xsimd::rotl(x[5], 12);
      x[6] = xsimd::rotl(x[6], 12);
      x[7] = xsimd::rotl(x[7], 12);
      x[4] = xsimd::rotl(x[4], 12);

      x[0] += x[5];
      x[1] += x[6];
      x[2] += x[7];
      x[3] += x[4];

      x[15] ^= x[0];
      x[12] ^= x[1];
      x[13] ^= x[2];
      x[14] ^= x[3];

      x[15] = xsimd::rotl(x[15], 8);
      x[12] = xsimd::rotl(x[12], 8);
      x[13] = xsimd::rotl(x[13], 8);
      x[14] = xsimd::rotl(x[14], 8);

      x[10] += x[15];
      x[11] += x[12];
      x[8] += x[13];
      x[9] += x[14];

      x[5] ^= x[10];
      x[6] ^= x[11];
      x[7] ^= x[8];
      x[4] ^= x[9];

      x[5] = xsimd::rotl(x[5], 7);
      x[6] = xsimd::rotl(x[6], 7);
      x[7] = xsimd::rotl(x[7], 7);
      x[4] = xsimd::rotl(x[4], 7);
    }

    for (auto i = 0; i < MATRIX_WORDCOUNT; ++i) {
      x[i] += simd_type::broadcast(m_state[i]);
    }
    x[12] += lower_counter_inc;
    x[13] += higher_counter_inc;


    // Batch i contains the i'th word of all chacha blocks, so transpose
    // to make row j have all words of the j'th chacha block.
    matrix_word tmp[SIMD_WIDTH];
    for (auto i = 0; i < MATRIX_WORDCOUNT; ++i) {
      x[i].store_unaligned(tmp);
      for (auto j = 0; j < SIMD_WIDTH; ++j) {
        m_cache[j][i] = tmp[j];
      }
    }

    m_state[12] += SIMD_WIDTH;
    m_state[13] += c < SIMD_WIDTH;
  }
};
}
