#include <memory>
#include <random/xoshiro_simd.hpp>
#include <xsimd/xsimd.hpp>

namespace prng {

using XoshiroSIMDCreator = internal::XoshiroSIMDCreator;

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX512F__)
template std::unique_ptr<XoshiroSIMD::IXoshiroSIMD> XoshiroSIMDCreator::operator()<xsimd::avx512f>(xsimd::avx512f) const;
#elif defined(__AVX__) && defined(__AVX2__) && defined(__FMA__)
template std::unique_ptr<XoshiroSIMD::IXoshiroSIMD> XoshiroSIMDCreator::operator()<xsimd::fma3<xsimd::avx2>>(xsimd::fma3<xsimd::avx2>) const;
#elif defined(__SSE4_1__) && defined(__SSE4_2__) && defined(__SSSE3__) && defined(__SSE3__)
template std::unique_ptr<XoshiroSIMD::IXoshiroSIMD> XoshiroSIMDCreator::operator()<xsimd::sse4_2>(xsimd::sse4_2) const;
#elif defined(__SSE__) && defined(__SSE2__) && defined(__MMX__)
template std::unique_ptr<XoshiroSIMD::IXoshiroSIMD> XoshiroSIMDCreator::operator()<xsimd::sse2>(xsimd::sse2) const;

#else
#error "no SIMD instruction set enabled"
#endif

#else
// here we can add mac and arm support
#error "Unsupported x86-64 architecture"

#endif

} // namespace prng
