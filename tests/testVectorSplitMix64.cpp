//
// Created by mbarbone on 5/9/23.
//

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "testVectorSplitMix64.cpp"  // this file
#include <hwy/foreach_target.h>                        // must come before highway.h included by splitmix
#include <hwy/highway.h>
#include <VectorXoshiro/vectorSplitMix64.h>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

HWY_BEFORE_NAMESPACE();    // required if not using HWY_ATTR

namespace HWY_NAMESPACE {  // required: unique per target

namespace hn = hwy::HWY_NAMESPACE;

void rngLoop(const std::uint64_t seed, std::uint64_t* HWY_RESTRICT result, const size_t size) {
    const hn::ScalableTag<std::uint64_t> d;
    VectorSplitMix64 rng{hn::Set(d, seed)};
    for (size_t i = 0; i < size; i += hn::Lanes(d)) {
        auto x = rng();
        hn::Store(x, d, result + i);
    }
}

}  // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();  // required if not using HWY_ATTR

#if HWY_ONCE


// This macro declares a static array used for dynamic dispatch.
HWY_EXPORT(rngLoop);

void CallMulRngLoop(std::uint64_t seed,  std::uint64_t* HWY_RESTRICT result, const size_t size) {
    // This must reside outside of HWY_NAMESPACE because it references (calls
    // the appropriate one from) the per-target implementations there. For
    // static dispatch, use HWY_STATIC_DISPATCH.
    return HWY_DYNAMIC_DISPATCH(rngLoop)(seed, result, size);
}

#endif
