#pragma once

#include "vectorXoshiro.hpp"

#include <array>
#include <cstdint>
#include <limits>

#include <xsimd/xsimd.hpp>

#include "xoshiro.hpp"

namespace xoshiro {

class VectorXoshiro;

namespace internal {

template <class Arch> class VectorXoshiroImpl {
public:
  using result_type = std::uint64_t;

protected:
  using simd_type = xsimd::batch<result_type, Arch>;
  static constexpr auto RNG_WIDTH = static_cast<std::uint_fast8_t>(4);
  static constexpr auto CACHE_SIZE = static_cast<std::uint_fast8_t>(64);
  static constexpr auto SIMD_WIDTH = static_cast<std::uint_fast8_t>(simd_type::size);

public:
  // Constructor: cache is provided externally by reference.
  __attribute__((noinline)) constexpr explicit VectorXoshiroImpl(const std::uint64_t seed,
                                                                 std::array<std::uint64_t, CACHE_SIZE> &cache) noexcept
      : m_cache(cache), m_state{}, m_index(CACHE_SIZE) {
    Xoshiro rng{seed};
    std::array<std::array<std::uint64_t, SIMD_WIDTH>, RNG_WIDTH> states{};
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

  // Additional constructor with thread_id; also requires a cache reference.
  constexpr explicit VectorXoshiroImpl(const std::uint64_t seed, const std::uint64_t thread_id,
                                       std::array<std::uint64_t, CACHE_SIZE> &cache) noexcept
      : VectorXoshiroImpl(seed, cache) {
    for (auto i = UINT8_C(0); i < thread_id; ++i) {
      jump();
    }
  }

  constexpr std::uint64_t operator()() noexcept {
    if (m_index == CACHE_SIZE) [[unlikely]] {
      populate_cache();
    }
    return m_cache[m_index++];
  }

  constexpr double uniform() noexcept { return static_cast<double>(operator()() >> 11) * 0x1.0p-53; }

  constexpr auto getState(const std::size_t index) const {
    std::array<std::uint64_t, RNG_WIDTH> state{};
    for (auto i = UINT8_C(0); i < RNG_WIDTH; ++i) {
      state[i] = m_state[i].get(index);
    }
    return state;
  }

  static constexpr result_type(min)() noexcept { return (std::numeric_limits<result_type>::min)(); }

  static constexpr result_type(max)() noexcept { return (std::numeric_limits<result_type>::max)(); }

  static constexpr std::uint64_t stateSize() noexcept { return RNG_WIDTH; }

private:
  // m_cache is now a reference; it must outlive this object.
  alignas(simd_type::arch_type::alignment()) std::array<std::uint64_t, CACHE_SIZE> &m_cache;
  std::array<simd_type, RNG_WIDTH> m_state;
  std::uint_fast8_t m_index;

  __always_inline constexpr auto next() noexcept {
    const auto result = rotl(m_state[0] + m_state[3], 23) + m_state[0];
    const auto t = m_state[1] << 17;

    m_state[2] ^= m_state[0];
    m_state[3] ^= m_state[1];
    m_state[1] ^= m_state[2];
    m_state[0] ^= m_state[3];

    m_state[2] ^= t;

    m_state[3] = rotl(m_state[3], 45);

    return result;
  }

  // Unrolled loop to populate the cache.
  template <size_t... Is> __always_inline constexpr void unroll_populate(std::index_sequence<Is...>) noexcept {
    ((next().store_aligned(m_cache.data() + Is * SIMD_WIDTH)), ...);
  }

  __attribute__((noinline)) constexpr void populate_cache() noexcept {
    unroll_populate(std::make_index_sequence<CACHE_SIZE / SIMD_WIDTH>{});
    m_index = 0;
  }

  friend VectorXoshiro;

public:
  /* This is the jump function for the generator. It is equivalent
     to 2^128 calls to next(); it can be used to generate 2^128
     non-overlapping subsequences for parallel computations. */
  constexpr void jump() noexcept {
    constexpr std::uint64_t JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c};
    for (auto _ = UINT8_C(0); _ < SIMD_WIDTH; ++_) {
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
    constexpr std::uint64_t LONG_JUMP[] = {0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241,
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

struct VectorXoshiroCreator;

} // namespace internal

class VectorXoshiroNative : public internal::VectorXoshiroImpl<xsimd::best_arch> {
public:
  using VectorXoshiroImpl::VectorXoshiroImpl;
  explicit VectorXoshiroNative(const result_type seed) noexcept : m_cache{}, VectorXoshiroImpl(seed, m_cache) {}

private:
  alignas(simd_type::arch_type::alignment()) std::array<result_type, CACHE_SIZE> m_cache;
};

class VectorXoshiro {
public:
  using result_type = internal::VectorXoshiroImpl<xsimd::best_arch>::result_type;
  explicit VectorXoshiro(std::uint64_t seed);
  __always_inline constexpr std::uint64_t operator()() noexcept {
    if (m_index == CACHE_SIZE) [[unlikely]] {
      pImpl->populate_cache();
      m_index = 0;
    }
    return m_cache[m_index++];
  }
  __always_inline constexpr double uniform() noexcept { return static_cast<double>(operator()() >> 11) * 0x1.0p-53; }
  __always_inline void jump() noexcept { pImpl->jump(); }
  __always_inline void long_jump() noexcept { pImpl->long_jump(); }

private:
  // Abstract interface to hide the templated implementation.
  static constexpr auto CACHE_SIZE = internal::VectorXoshiroImpl<xsimd::default_arch>::CACHE_SIZE;

  struct IVectorXoshiro {
    virtual ~IVectorXoshiro() = default;
    virtual void populate_cache() noexcept = 0;
    virtual void jump() noexcept = 0;
    virtual void long_jump() noexcept = 0;
  };

  // Templated wrapper that delegates to internal::VectorXoshiroImpl<Arch>.
  template <class Arch> class ImplWrapper : public IVectorXoshiro {
    internal::VectorXoshiroImpl<Arch> impl;

  public:
    explicit ImplWrapper(std::uint64_t seed, std::array<result_type, CACHE_SIZE> &cache) : impl(seed, cache) {}
    __always_inline void populate_cache() noexcept final { impl.populate_cache(); }
    __always_inline void jump() noexcept final { impl.jump(); }
    __always_inline void long_jump() noexcept final { impl.long_jump(); }
  };

  alignas(xsimd::avx512f::alignment()) std::array<result_type, CACHE_SIZE> m_cache;
  std::unique_ptr<IVectorXoshiro> pImpl;
  std::uint_fast8_t m_index;
  // Friend declaration for the external factory function.
  friend std::unique_ptr<IVectorXoshiro> create_vector_xoshiro_impl(std::uint64_t seed, std::array<result_type, CACHE_SIZE> & cache);
  friend internal::VectorXoshiroCreator;
};

// Extern function declaration.
std::unique_ptr<VectorXoshiro::IVectorXoshiro> create_vector_xoshiro_impl(std::uint64_t seed, std::array<VectorXoshiro::result_type, VectorXoshiro::CACHE_SIZE>& cache);

namespace internal {

// Functor used by xsimd::dispatch.
// It receives an architecture tag and returns a pointer to the corresponding
// implementation.
struct VectorXoshiroCreator {
  std::uint64_t seed;
  std::array<VectorXoshiro::result_type, VectorXoshiro::CACHE_SIZE> &cache;
  template <class Arch> std::unique_ptr<VectorXoshiro::IVectorXoshiro> operator()(Arch) const;
};

template <class Arch> std::unique_ptr<VectorXoshiro::IVectorXoshiro> VectorXoshiroCreator::operator()(Arch) const {
  return std::make_unique<VectorXoshiro::ImplWrapper<Arch>>(seed, cache);
}

extern template std::unique_ptr<VectorXoshiro::IVectorXoshiro>
VectorXoshiroCreator::operator()<xsimd::sse2>(xsimd::sse2) const;
extern template std::unique_ptr<VectorXoshiro::IVectorXoshiro>
VectorXoshiroCreator::operator()<xsimd::sse4_2>(xsimd::sse4_2) const;
extern template std::unique_ptr<VectorXoshiro::IVectorXoshiro>
VectorXoshiroCreator::operator()<xsimd::avx2>(xsimd::avx2) const;
extern template std::unique_ptr<VectorXoshiro::IVectorXoshiro>
VectorXoshiroCreator::operator()<xsimd::avx512f>(xsimd::avx512f) const;

} // namespace internal

} // namespace xoshiro
