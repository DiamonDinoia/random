#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <xsimd/xsimd.hpp>

#include "random/macros.hpp"
#include "xsimd/types/xsimd_api.hpp"

namespace prng {

template <std::uint8_t R, class Arch>
class ChaChaSIMD {
protected:
  static constexpr auto MATRIX_ROW_LEN = std::uint8_t{4};
  static constexpr auto MATRIX_COL_LEN = std::uint8_t{4};
  static constexpr auto MATRIX_WORDCOUNT = std::uint8_t{16};
  static constexpr auto KEY_WORDCOUNT = std::uint8_t{8};

public:
  using input_word = std::uint64_t;
  using matrix_word = std::uint32_t;
  using matrix_type = std::array<matrix_word, MATRIX_WORDCOUNT>;
  using rounds_type = std::uint8_t;
  using simd_type = xsimd::batch<matrix_word, Arch>;

protected:
  static constexpr std::uint8_t SIMD_WIDTH = std::uint8_t{simd_type::size};
  static constexpr auto CACHE_WORDCOUNT = std::uint16_t{MATRIX_WORDCOUNT * SIMD_WIDTH};
  static constexpr auto CACHE_BLOCKCOUNT = SIMD_WIDTH;

public:
  /**
   * @brief Construct a SIMD ChaCha generator with given key, counter and nonce
   * @param key A 256-bit key, divided up into eight 32-bit words.
   * @param counter Initial value of the counter.
   * @param nonce Initial value of the nonce.
   */
  explicit PRNG_ALWAYS_INLINE ChaChaSIMD(
    const std::array<matrix_word, KEY_WORDCOUNT> key,
    const input_word counter,
    const input_word nonce
  ) {
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

  /**
   * @brief Generates the next random block.
   * @return The next random block.
   */
  PRNG_ALWAYS_INLINE constexpr matrix_type operator()() noexcept {
    if (m_index >= CACHE_BLOCKCOUNT) [[unlikely]] {
      gen_next_blocks_in_cache();
      m_index = 0;
    }
    matrix_word *cache_block = m_cache.data() + (m_index++ * MATRIX_WORDCOUNT);
    matrix_type block;
    std::copy(cache_block, cache_block + MATRIX_WORDCOUNT, block.begin());
    return block;
  }

  /**
   * @brief Returns the state of the generator; a 4x4 matrix.
   * @return State of the generator.
   */
  PRNG_ALWAYS_INLINE constexpr matrix_type getState() const noexcept { return m_state; }


private:
  matrix_type m_state;
  std::array<matrix_word, CACHE_WORDCOUNT> m_cache;
  // Initalize to "past end of the cache" since cache starts empty
  std::uint8_t m_index = CACHE_BLOCKCOUNT;

  /**
   * Return an array { 0, 1 * step, ..., (n - 1) * step }. Can be used to initialize a batch for
   * consecutively incremeting elements in a batch of low counter words, as well as
   * a batch of offsets to scatter matrix words into memory with.
  */
  template <size_t... Is>
  static constexpr PRNG_ALWAYS_INLINE std::array<matrix_word, sizeof...(Is)> matrix_word_sequence(std::index_sequence<Is...>, std::uint8_t step = 1) noexcept {
    return {static_cast<matrix_word>(Is * step)...};
  }

  /**
   * Return an array { 0, n < 1, n < 2, ..., n < (SIMD_WIDTH - 1) }. Can be used to initialize a
   * batch for incremetinng elements in a batch of consecutive high counter words, depending on at
   * what index the corresponding lower counter words had an overflow.
   */
  PRNG_ALWAYS_INLINE static constexpr std::array<matrix_word, SIMD_WIDTH> make_higher_counter_inc(std::uint8_t n) noexcept {
    std::array<matrix_word, SIMD_WIDTH> incs;
    incs[0] = 0;
    for (auto i = 1; i < SIMD_WIDTH; ++i) {
      incs[i] = n < i;
    }
    return incs;
  }

  /**
   * Generates `SIMD_WIDTH` new ChaCha blocks, and write them all into the cache. Will overwrite
   * anything else in the cache.
   */
  PRNG_ALWAYS_INLINE constexpr void gen_next_blocks_in_cache() noexcept {
    alignas(simd_type::arch_type::alignment()) simd_type lower_counter_inc, higher_counter_inc;
    lower_counter_inc =
      xsimd::load_aligned(matrix_word_sequence(std::make_index_sequence<SIMD_WIDTH>{}).data());
    matrix_word overflow_index = std::numeric_limits<matrix_word>::max() - m_state[12];
    if (overflow_index < SIMD_WIDTH) [[unlikely]] {
      higher_counter_inc = xsimd::load_aligned(make_higher_counter_inc(overflow_index).data());
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

      x[12] = xsimd::rotl<16>(x[12]);
      x[13] = xsimd::rotl<16>(x[13]);
      x[14] = xsimd::rotl<16>(x[14]);
      x[15] = xsimd::rotl<16>(x[15]);

      x[8] += x[12];
      x[9] += x[13];
      x[10] += x[14];
      x[11] += x[15];

      x[4] ^= x[8];
      x[5] ^= x[9];
      x[6] ^= x[10];
      x[7] ^= x[11];

      x[4] = xsimd::rotl<12>(x[4]);
      x[5] = xsimd::rotl<12>(x[5]);
      x[6] = xsimd::rotl<12>(x[6]);
      x[7] = xsimd::rotl<12>(x[7]);

      x[0] += x[4];
      x[1] += x[5];
      x[2] += x[6];
      x[3] += x[7];

      x[12] ^= x[0];
      x[13] ^= x[1];
      x[14] ^= x[2];
      x[15] ^= x[3];

      x[12] = xsimd::rotl<8>(x[12]);
      x[13] = xsimd::rotl<8>(x[13]);
      x[14] = xsimd::rotl<8>(x[14]);
      x[15] = xsimd::rotl<8>(x[15]);

      x[8] += x[12];
      x[9] += x[13];
      x[10] += x[14];
      x[11] += x[15];

      x[4] ^= x[8];
      x[5] ^= x[9];
      x[6] ^= x[10];
      x[7] ^= x[11];

      x[4] = xsimd::rotl<7>(x[4]);
      x[5] = xsimd::rotl<7>(x[5]);
      x[6] = xsimd::rotl<7>(x[6]);
      x[7] = xsimd::rotl<7>(x[7]);

      x[0] += x[5];
      x[1] += x[6];
      x[2] += x[7];
      x[3] += x[4];

      x[15] ^= x[0];
      x[12] ^= x[1];
      x[13] ^= x[2];
      x[14] ^= x[3];

      x[15] = xsimd::rotl<16>(x[15]);
      x[12] = xsimd::rotl<16>(x[12]);
      x[13] = xsimd::rotl<16>(x[13]);
      x[14] = xsimd::rotl<16>(x[14]);

      x[10] += x[15];
      x[11] += x[12];
      x[8] += x[13];
      x[9] += x[14];

      x[5] ^= x[10];
      x[6] ^= x[11];
      x[7] ^= x[8];
      x[4] ^= x[9];

      x[5] = xsimd::rotl<12>(x[5]);
      x[6] = xsimd::rotl<12>(x[6]);
      x[7] = xsimd::rotl<12>(x[7]);
      x[4] = xsimd::rotl<12>(x[4]);

      x[0] += x[5];
      x[1] += x[6];
      x[2] += x[7];
      x[3] += x[4];

      x[15] ^= x[0];
      x[12] ^= x[1];
      x[13] ^= x[2];
      x[14] ^= x[3];

      x[15] = xsimd::rotl<8>(x[15]);
      x[12] = xsimd::rotl<8>(x[12]);
      x[13] = xsimd::rotl<8>(x[13]);
      x[14] = xsimd::rotl<8>(x[14]);

      x[10] += x[15];
      x[11] += x[12];
      x[8] += x[13];
      x[9] += x[14];

      x[5] ^= x[10];
      x[6] ^= x[11];
      x[7] ^= x[8];
      x[4] ^= x[9];

      x[5] = xsimd::rotl<7>(x[5]);
      x[6] = xsimd::rotl<7>(x[6]);
      x[7] = xsimd::rotl<7>(x[7]);
      x[4] = xsimd::rotl<7>(x[4]);
    }

    for (auto i = 0; i < MATRIX_WORDCOUNT; ++i) {
      x[i] += simd_type::broadcast(m_state[i]);
    }
    // Remember to apply counter increments when summing rounds results with the original states.
    x[12] += lower_counter_inc;
    x[13] += higher_counter_inc;

    // Batch i contains the i'th word of all chacha blocks, so transpose to get rows of chacha blocks.
    alignas(simd_type::arch_type::alignment()) simd_type scatter_offsets =
      xsimd::load_aligned(matrix_word_sequence(std::make_index_sequence<SIMD_WIDTH>{}, MATRIX_WORDCOUNT).data());
    for (auto i = 0; i < MATRIX_WORDCOUNT; ++i) {
      x[i].scatter(m_cache.data() + i, scatter_offsets);
    }

    m_state[12] += SIMD_WIDTH;
    m_state[13] += overflow_index < SIMD_WIDTH;
  }
};
}
