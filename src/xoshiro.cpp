#include "xoshiro/vectorXoshiro.hpp"

namespace xoshiro {

using namespace internal;


std::unique_ptr<VectorXoshiro::IVectorXoshiro> create_vector_xoshiro_impl(std::uint64_t seed) {
  // Dispatch to the appropriate implementation based on runtime-detected architecture.
  static auto dispatch = xsimd::dispatch<xsimd::arch_list<xsimd::avx512f, xsimd::avx2, xsimd::sse4_2, xsimd::sse2>>(VectorXoshiroCreator{seed});
  return dispatch();
}

// Definitions of VectorXoshiro's member functions.
VectorXoshiro::VectorXoshiro(std::uint64_t seed)
    : pImpl(create_vector_xoshiro_impl(seed))
{ }

} // namespace xoshiro