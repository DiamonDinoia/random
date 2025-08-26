/*  Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

Ported to C++, vectorized, and optimized by Marco Barbone.
Original implementation by David Blackman and Sebastiano
Vigna.
*/

#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include <xsimd/xsimd.hpp>

#include "macros.hpp"
#include "xoshiro_scalar.hpp"

namespace prng {

/**
 * Forward declaration of the XoshiroSIMD class.
 */
class XoshiroSIMD;

namespace internal {

/**
 * Implementation of the XoshiroSIMD class template.
 *
 * @tparam Arch The architecture type for SIMD operations.
 */
template <class Arch> class XoshiroSIMDImpl {
public:
  using result_type = std::uint64_t;
  static constexpr PRNG_ALWAYS_INLINE auto(min)() noexcept { return (std::numeric_limits<result_type>::min)(); }
  static constexpr PRNG_ALWAYS_INLINE auto(max)() noexcept { return (std::numeric_limits<result_type>::max)(); }
  static constexpr PRNG_ALWAYS_INLINE auto stateSize() noexcept { return RNG_WIDTH; }

protected:
  using simd_type = xsimd::batch<result_type, Arch>;
  static constexpr auto RNG_WIDTH = std::uint8_t{4};
  static constexpr auto SIMD_WIDTH = std::uint8_t{simd_type::size};
  static constexpr auto CACHE_SIZE = std::uint16_t{std::numeric_limits<std::uint8_t>::max() + 1};

public:
  /**
   * Constructor that initializes the generator with a seed and an external cache.
   *
   * @param seed The seed value.
   * @param cache Reference to the external cache.
   */
  PRNG_ALWAYS_INLINE constexpr explicit XoshiroSIMDImpl(const result_type seed,
                                                        std::array<result_type, CACHE_SIZE> &cache) noexcept
      : m_cache(cache), m_state{}, m_index{0} {
    XoshiroScalar rng{seed};
    std::array<std::array<result_type, SIMD_WIDTH>, RNG_WIDTH> states{};
    for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
      for (auto j = 0UL; j < RNG_WIDTH; ++j) {
        states[j][i] = rng.getState()[j];
      }
      rng.jump();
    }
    for (auto i = UINT8_C(0); i < RNG_WIDTH; ++i) {
      m_state[i] = simd_type::load_unaligned(states[i].data());
    }
  }

  /**
   * Constructor that initializes the generator with a seed, thread ID, and an external cache.
   *
   * @param seed The seed value.
   * @param thread_id The thread ID.
   * @param cache Reference to the external cache.
   */
  PRNG_ALWAYS_INLINE constexpr explicit XoshiroSIMDImpl(const result_type seed, const result_type thread_id,
                                                        std::array<result_type, CACHE_SIZE> &cache) noexcept
      : XoshiroSIMDImpl(seed, cache) {
    for (result_type i = 0; i < thread_id; ++i) {
      jump();
    }
  }

  /**
   * Constructor that initializes the generator with a seed, thread ID, and an external cache.
   *
   * @param seed The seed value.
   * @param thread_id The thread ID.
   * @param cluster_id The cluster ID.
   * @param cache Reference to the external cache.
   */
  PRNG_ALWAYS_INLINE constexpr explicit XoshiroSIMDImpl(const result_type seed, const result_type thread_id,
                                                        const result_type cluster_id,
                                                        std::array<result_type, CACHE_SIZE> &cache) noexcept
      : XoshiroSIMDImpl(seed, thread_id, cache) {
    for (result_type i = 0; i < cluster_id; ++i) {
      long_jump();
    }
  }

  /**
   * Generates the next random number.
   *
   * @return The next random number.
   */
  PRNG_ALWAYS_INLINE constexpr auto operator()() noexcept {
    if (m_index == 0) {
      populate_cache();
    }
    return m_cache[m_index++];
  }

  /**
   * Generates a uniform random number in the range [0, 1).
   *
   * @return A uniform random number.
   */
  PRNG_ALWAYS_INLINE constexpr auto uniform() noexcept { return static_cast<double>(operator()() >> 11) * 0x1.0p-53; }

  /**
   * Returns the state of the generator at the specified index.
   *
   * @param index The index of the state.
   * @return The state at the specified index.
   */
  PRNG_ALWAYS_INLINE constexpr auto getState(const std::size_t index) const noexcept {
    std::array<result_type, RNG_WIDTH> state{};
    for (auto i = UINT8_C(0); i < RNG_WIDTH; ++i) {
      state[i] = m_state[i].get(index);
    }
    return state;
  }

  /**
   * Jump function for the generator. It is equivalent to 2^128 calls to next().
   * It can be used to generate 2^128 non-overlapping subsequences for parallel computations.
   */
  PRNG_ALWAYS_INLINE constexpr void jump() noexcept {
    constexpr result_type JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c};
    for (auto _ = UINT8_C(0); _ < SIMD_WIDTH; ++_) {
      simd_type s0(0);
      simd_type s1(0);
      simd_type s2(0);
      simd_type s3(0);
      for (const auto i : JUMP)
        for (auto b = 0; b < 64; b++) {
          if (i & result_type{1} << b) {
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

  /**
   * Long-jump function for the generator. It is equivalent to 2^192 calls to next().
   * It can be used to generate 2^64 starting points, from each of which jump() will generate 2^64 non-overlapping
   * subsequences for parallel distributed computations.
   */
  PRNG_ALWAYS_INLINE constexpr void long_jump() noexcept {
    constexpr result_type LONG_JUMP[] = {0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241,
                                         0x39109bb02acbe635};
    simd_type s0(0);
    simd_type s1(0);
    simd_type s2(0);
    simd_type s3(0);
    for (const auto i : LONG_JUMP)
      for (auto b = 0; b < 64; b++) {
        if (i & result_type{1} << b) {
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

private:
  alignas(simd_type::arch_type::alignment()) std::array<result_type, CACHE_SIZE> &m_cache;
  std::array<simd_type, RNG_WIDTH> m_state;
  std::uint8_t m_index;

  /**
   * Generates the next state of the generator.
   *
   * @return The next state.
   */
  PRNG_ALWAYS_INLINE constexpr auto next() noexcept {
    const auto result = xsimd::rotl(m_state[0] + m_state[3], 23) + m_state[0];
    const auto t = m_state[1] << 17;

    m_state[2] ^= m_state[0];
    m_state[3] ^= m_state[1];
    m_state[1] ^= m_state[2];
    m_state[0] ^= m_state[3];

    m_state[2] ^= t;

    m_state[3] = xsimd::rotl(m_state[3], 45);

    return result;
  }

  /**
   * Unrolled loop to populate the cache.
   *
   * @tparam Is The indices of the cache.
   */
  template <size_t... Is> PRNG_ALWAYS_INLINE constexpr void unroll_populate(std::index_sequence<Is...>) noexcept {
    (next().store_aligned(m_cache.data() + Is * SIMD_WIDTH), ...);
  }

  /**
   * Populates the cache with random numbers.
   */
  PRNG_ALWAYS_INLINE constexpr void populate_cache() noexcept {
    unroll_populate(std::make_index_sequence<CACHE_SIZE / SIMD_WIDTH>{});
  }

  friend XoshiroSIMD;
};

struct XoshiroSIMDCreator;

} // namespace internal

/**
 * XoshiroSIMD class using the best available architecture.
 */
class XoshiroNative : public internal::XoshiroSIMDImpl<xsimd::best_arch> {
public:
  using XoshiroSIMDImpl::XoshiroSIMDImpl;

  /**
   * Constructor that initializes the generator with a seed.
   *
   * @param seed The seed value.
   */
  PRNG_ALWAYS_INLINE explicit XoshiroNative(const result_type seed) noexcept : XoshiroSIMDImpl(seed, m_cache) {}
  PRNG_ALWAYS_INLINE explicit XoshiroNative(const result_type seed, const result_type thread_id) noexcept
      : XoshiroSIMDImpl(seed, thread_id, m_cache) {}
  PRNG_ALWAYS_INLINE explicit XoshiroNative(const result_type seed, const result_type thread_id,
                                            const result_type cluster_id) noexcept
      : XoshiroSIMDImpl(seed, thread_id, cluster_id, m_cache) {}

private:
  alignas(simd_type::arch_type::alignment()) std::array<result_type, CACHE_SIZE> m_cache{};
};

/**
 * XoshiroSIMD class that provides a high-level interface for the generator.
 */
class XoshiroSIMD {
public:
  using result_type = internal::XoshiroSIMDImpl<xsimd::best_arch>::result_type;
  constexpr static PRNG_ALWAYS_INLINE result_type(min)() noexcept {
    return internal::XoshiroSIMDImpl<xsimd::best_arch>::min();
  }
  constexpr static PRNG_ALWAYS_INLINE result_type(max)() noexcept {
    return internal::XoshiroSIMDImpl<xsimd::best_arch>::max();
  }

  explicit XoshiroSIMD(result_type seed, result_type thread_id = 0, result_type cluster_id = 0) noexcept;

  /**
   * Generates the next random number.
   *
   * @return The next random number.
   */
  PRNG_ALWAYS_INLINE result_type operator()() noexcept {
    if (m_index == 0) [[unlikely]] {
      pImpl->populate_cache();
    }
    return m_cache[m_index++];
  }

  /**
   * Generates a uniform random number in the range [0, 1).
   *
   * @return A uniform random number.
   */
  PRNG_ALWAYS_INLINE double uniform() noexcept { return static_cast<double>(operator()() >> 11) * 0x1.0p-53; }

  /**
   * Jump function for the generator.
   */
  PRNG_ALWAYS_INLINE void jump() noexcept { pImpl->jump(); }

  /**
   * Long-jump function for the generator.
   */
  PRNG_ALWAYS_INLINE void long_jump() noexcept { pImpl->long_jump(); }

private:
  static constexpr auto CACHE_SIZE = internal::XoshiroSIMDImpl<xsimd::default_arch>::CACHE_SIZE;

  /**
   * Abstract interface to hide the templated implementation.
   */
  struct IXoshiroSIMD {
    virtual ~IXoshiroSIMD() = default;
    virtual void populate_cache() noexcept = 0;
    virtual void jump() noexcept = 0;
    virtual void long_jump() noexcept = 0;
  };

  /**
   * Templated wrapper that delegates to internal::XoshiroSIMDImpl<Arch>.
   *
   * @tparam Arch The architecture type for SIMD operations.
   */
  template <class Arch> class ImplWrapper : public IXoshiroSIMD {
    internal::XoshiroSIMDImpl<Arch> impl;

  public:
    PRNG_ALWAYS_INLINE explicit ImplWrapper(result_type seed, result_type thread_id, result_type cluster_id,
                                            std::array<result_type, CACHE_SIZE> &cache) noexcept
        : impl(seed, thread_id, cluster_id, cache) {}
    PRNG_ALWAYS_INLINE void populate_cache() noexcept final { impl.populate_cache(); }
    PRNG_ALWAYS_INLINE void jump() noexcept final { impl.jump(); }
    PRNG_ALWAYS_INLINE void long_jump() noexcept final { impl.long_jump(); }
  };

  alignas(xsimd::avx512f::alignment()) std::array<result_type, CACHE_SIZE> m_cache;
  std::unique_ptr<IXoshiroSIMD> pImpl;
  std::uint8_t m_index;

  friend std::unique_ptr<IXoshiroSIMD> create_xoshiro_simd_impl(result_type seed, result_type thread_id,
                                                                result_type cluster_id,
                                                                std::array<result_type, CACHE_SIZE> &cache);
  friend internal::XoshiroSIMDCreator;
};

/**
 * Extern function declaration to create a XoshiroSIMD implementation.
 *
 * @param seed The seed value.
 * @param thread_id The thread ID.
 * @param cluster_id The cluster ID.
 * @param cache Reference to the external cache.
 * @return A unique pointer to the XoshiroSIMD implementation.
 */
std::unique_ptr<XoshiroSIMD::IXoshiroSIMD>
create_xoshiro_simd_impl(XoshiroSIMD::result_type seed, XoshiroSIMD::result_type thread_id,
                         XoshiroSIMD::result_type cluster_id,
                         std::array<XoshiroSIMD::result_type, XoshiroSIMD::CACHE_SIZE> &cache);

namespace internal {

/**
 * Functor used by xsimd::dispatch to create a XoshiroSIMD implementation.
 */
struct XoshiroSIMDCreator {
  XoshiroSIMD::result_type seed, thread_id, cluster_id;
  std::array<XoshiroSIMD::result_type, XoshiroSIMD::CACHE_SIZE> &cache;

  /**
   * Operator that creates a XoshiroSIMD implementation for the given architecture.
   *
   * @tparam Arch The architecture type for SIMD operations.
   * @param arch The architecture tag.
   * @return A unique pointer to the XoshiroSIMD implementation.
   */
  template <class Arch> std::unique_ptr<XoshiroSIMD::IXoshiroSIMD> operator()(Arch) const;
};

template <class Arch> std::unique_ptr<XoshiroSIMD::IXoshiroSIMD> XoshiroSIMDCreator::operator()(Arch) const {
  return std::make_unique<XoshiroSIMD::ImplWrapper<Arch>>(seed, thread_id, cluster_id, cache);
}

extern template std::unique_ptr<XoshiroSIMD::IXoshiroSIMD>
XoshiroSIMDCreator::operator()<xsimd::sse2>(xsimd::sse2) const;
extern template std::unique_ptr<XoshiroSIMD::IXoshiroSIMD>
XoshiroSIMDCreator::operator()<xsimd::sse4_2>(xsimd::sse4_2) const;
extern template std::unique_ptr<XoshiroSIMD::IXoshiroSIMD>
XoshiroSIMDCreator::operator()<xsimd::fma3<xsimd::avx2>>(xsimd::fma3<xsimd::avx2>) const;
extern template std::unique_ptr<XoshiroSIMD::IXoshiroSIMD>
XoshiroSIMDCreator::operator()<xsimd::avx512f>(xsimd::avx512f) const;

} // namespace internal

} // namespace prng
