#include "xoshiro/vectorXoshiro.hpp"

namespace xoshiro {

using namespace internal;

std::unique_ptr<VectorXoshiro::IVectorXoshiro>
create_vector_xoshiro_impl(VectorXoshiro::result_type seed, VectorXoshiro::result_type thread_id,
                           VectorXoshiro::result_type cluster_id,
                           std::array<VectorXoshiro::result_type, VectorXoshiro::CACHE_SIZE> &cache) {
  // Dispatch to the appropriate implementation based on runtime-detected
  // architecture.
  return xsimd::dispatch<xsimd::arch_list<xsimd::avx512f, xsimd::fma3<xsimd::avx2>, xsimd::sse4_2, xsimd::sse2>>(
      VectorXoshiroCreator{seed, thread_id, cluster_id, cache})();
}

// Definitions of VectorXoshiro's member functions.
VectorXoshiro::VectorXoshiro(const result_type seed, const result_type thread_id, const result_type cluster_id)
    : m_cache{}, pImpl{create_vector_xoshiro_impl(seed, thread_id, cluster_id, m_cache)}, m_index{0} {}
} // namespace xoshiro