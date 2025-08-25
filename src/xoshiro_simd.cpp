#include "random/xoshiro_simd.hpp"

namespace prng {

using namespace internal;

std::unique_ptr<XoshiroSIMD::IXoshiroSIMD>
create_xoshiro_simd_impl(XoshiroSIMD::result_type seed, XoshiroSIMD::result_type thread_id,
                         XoshiroSIMD::result_type cluster_id,
                         std::array<XoshiroSIMD::result_type, XoshiroSIMD::CACHE_SIZE> &cache) {
  // Dispatch to the appropriate implementation based on runtime-detected
  // architecture.
  return xsimd::dispatch<xsimd::arch_list<xsimd::avx512f, xsimd::fma3<xsimd::avx2>, xsimd::sse4_2, xsimd::sse2>>(
      XoshiroSIMDCreator{seed, thread_id, cluster_id, cache})();
}

// Definitions of XoshiroSIMD's member functions.
XoshiroSIMD::XoshiroSIMD(const result_type seed, const result_type thread_id, const result_type cluster_id) noexcept
    : m_cache{}, pImpl{create_xoshiro_simd_impl(seed, thread_id, cluster_id, m_cache)}, m_index{0} {}
} // namespace prng
