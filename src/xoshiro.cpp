#include "xoshiro/vectorXoshiro.hpp"

namespace xoshiro {

using namespace internal;

std::unique_ptr<VectorXoshiro::IVectorXoshiro> create_vector_xoshiro_impl(std::uint64_t seed, std::array<VectorXoshiro::result_type, VectorXoshiro::CACHE_SIZE>& cache) {
  // Dispatch to the appropriate implementation based on runtime-detected
  // architecture.
  static auto dispatch = xsimd::dispatch<xsimd::arch_list<xsimd::avx512f, xsimd::avx2, xsimd::sse4_2, xsimd::sse2>>(
      VectorXoshiroCreator{seed, cache});
  return dispatch();
}

// Definitions of VectorXoshiro's member functions.
VectorXoshiro::VectorXoshiro(std::uint64_t seed)
    : m_cache{}, pImpl{create_vector_xoshiro_impl(seed, m_cache)}, m_index{CACHE_SIZE} {}

} // namespace xoshiro