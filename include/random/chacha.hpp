#pragma once

#include <cstdint> 

#include "macros.hpp"

#if __cplusplus >= 202002L
#include <bit>
#endif

namespace prng {

template<std::uint8_t R = 20>
class ChaCha {

protected:
  static constexpr auto MATRIX_WORDCOUNT = std::uint8_t{16};
  static constexpr auto KEY_WORDCOUNT = std::uint8_t{8};

public:
  using input_word = std::uint64_t;
  using matrix_word = std::uint32_t;
  using matrix_type = std::array<matrix_word, MATRIX_WORDCOUNT>;

  // NOTE: We could perahps instead opt to pass a whole state as a singular input argument
  // That'd also enable us to make this constexpr but I digress.
  /**
   * @brief Construct the ChaChaScalar generator with given key, counter and nonce
   * @param key The key, assumes bytes are in little endian order.
   * @param counter Initial value of the counter.
   * @param nonce Initial value of the nonce.
   */
  PRNG_ALWAYS_INLINE explicit ChaCha(
    const std::array<matrix_word, KEY_WORDCOUNT> key,
    const input_word counter,
    const input_word nonce
  ) noexcept {
    // First four words (i.e. top-row) are always the same constants
    // They spell out "expand 2-byte k" in ASCII (little-endian)
    m_state[0] = 0x61707865;
    m_state[1] = 0x3320646e;
    m_state[2] = 0x79622d32;
    m_state[3] = 0x6b206574;

    for (int i = 0; i < 8; ++i) {
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
  PRNG_ALWAYS_INLINE constexpr matrix_type(operator())() noexcept { return next(); }

  /**
   * @brief Returns the state of the generator; a 4x4 matrix.
   * @return State of the generator.
   */
  PRNG_ALWAYS_INLINE constexpr matrix_type getState() const noexcept { return m_state; }

private:
  matrix_type m_state;

  // NOTE: THis is almost identical to the rotl in xoshiro_scalar, safe for the nubmer
  // of bits the operation is performed on. Maybe wanna consider making it a shared func?
  /**
   * @brief Rotates the bits of a 32-bit integer to the left.
   * @param x The integer to rotate.
   * @param k The number of bits to rotate.
   * @return The rotated integer.
   */
  static constexpr PRNG_ALWAYS_INLINE auto rotl(const matrix_word x, const int k) noexcept {
#if __cplusplus >= 202002L
    return std::rotl(x, k);
#else
    return (x << k) | (x >> (32 - k));
#endif
};

  /**
   * @brief Perform a quarter round (as defined for ChaCha stream ciphers) on a given matrix.
   * @param matrix The matrix to perform the quarter round on.
   * @param a First matrix index to perform quarter round on.
   * @param b Second matrix index to perform quarter round on.
   * @param c Third matrix index to perform quarter round on.
   * @param d Fourht matrix index to perform quarter round on.
   */
  static constexpr PRNG_ALWAYS_INLINE void quarter_round(
    matrix_type &m,
    const unsigned int a,
    const unsigned int b,
    const unsigned int c,
    const unsigned int d
  ) noexcept {
    m[a] += m[b]; m[d] ^= m[a]; m[d] = rotl(m[d], 16);
    m[c] += m[d]; m[b] ^= m[c]; m[b] = rotl(m[b], 12);
    m[a] += m[b]; m[d] ^= m[a]; m[d] = rotl(m[d],  8);
    m[c] += m[d]; m[b] ^= m[c]; m[b] = rotl(m[b],  7);
  }

  /**
   * @brief Increments the counter component of the state by 1
   */
  constexpr PRNG_ALWAYS_INLINE void inc_counter() noexcept {
    const matrix_word lower = m_state[12];
    const matrix_word upper = m_state[13];
    input_word counter = (static_cast<input_word>(upper) << 32) | static_cast<input_word>(lower);
    ++counter;
    m_state[12] = static_cast<matrix_word>(counter & 0xFFFFFFFF);
    m_state[13] = static_cast<matrix_word>(counter >> 32);
  }

  /**
   * @brief Returns the next output from the generator, then increases state's counter by 1.
   * @return The output for the current internal state.
  */
  constexpr PRNG_ALWAYS_INLINE matrix_type next() noexcept {
    matrix_type x = m_state;
    
    // Note that we perform both an odd and even round at the same time.
    // As a result the amount of rounds performed is always rounded up to an even number.
    for (auto i = 0; i < R; i += 2) {
      // Odd round
      quarter_round(x, 0, 4, 8,12);
      quarter_round(x, 1, 5, 9,13);
      quarter_round(x, 2, 6,10,14);
      quarter_round(x, 3, 7,11,15);

      // Even round
      quarter_round(x, 0, 5,10,15);
      quarter_round(x, 1, 6,11,12);
      quarter_round(x, 2, 7, 8,13);
      quarter_round(x, 3, 4, 9,14);
    }

    for (int i = 0; i < 16; ++i) {
      x[i] += m_state[i];
    }

    inc_counter();

    return x;
  }
};

}
