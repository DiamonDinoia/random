#include "xoshiro/vectorXoshiro.hpp"

namespace xoshiro {

using namespace internal;


std::unique_ptr<VectorXoshiro::ImplVariant> VectorXoshiro::create_vector_xoshiro_impl(const std::uint64_t seed) {
  // Dispatch to the appropriate implementation based on runtime-detected architecture.
  return xsimd::dispatch<xsimd::arch_list<xsimd::avx512f, xsimd::avx2, xsimd::sse4_2, xsimd::sse2>>(VectorXoshiroCreator{seed})();
}

// Definitions of VectorXoshiro's member functions.
VectorXoshiro::VectorXoshiro(const std::uint64_t seed)
    : impl_(create_vector_xoshiro_impl(seed)){}

} // namespace xoshiro