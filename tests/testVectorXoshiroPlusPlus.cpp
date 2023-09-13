//
// Created by mbarbone on 5/20/23.
//

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "testVectorXoshiroPlusPlus.cpp"  // this file
#include <VectorXoshiro/vectorXoshiroPlusPlus.h>
#include <hwy/foreach_target.h>                               // must come before highway.h included by splitmix
#include <hwy/highway.h>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>


HWY_BEFORE_NAMESPACE();    // required if not using HWY_ATTR

namespace HWY_NAMESPACE {  // required: unique per target

constexpr static auto tests = 1UL<<15;

namespace hn = hwy::HWY_NAMESPACE;

void rngLoop(const std::uint64_t seed, std::uint64_t* HWY_RESTRICT result, const size_t size) {
    const hn::ScalableTag<std::uint64_t> d;
    using ARRAY_T = decltype(hn::Undefined(hn::ScalableTag<std::uint64_t>()));

    vectorXoshiroPlusPlus<ARRAY_T> gernerator{seed};
    for (size_t i = 0; i < size; i += hn::Lanes(d)) {
        auto x = gernerator();
        hn::Store(x, d, result + i);
    }
}


void uniformLoop(const std::uint64_t seed, double* HWY_RESTRICT result, const size_t size) {
    const hn::ScalableTag<std::uint64_t> d;
    using ARRAY_T = decltype(hn::Undefined(hn::ScalableTag<std::uint64_t>()));
    vectorXoshiroPlusPlus<ARRAY_T> gernerator{seed};
    for (size_t i = 0; i < size; i += hn::Lanes(d)) {
        auto x = gernerator.uniform();
        hn::Store(x, d, result + i);
    }
}

void test_seeding(const std::uint64_t seed){
    const hn::ScalableTag<std::uint64_t> d;
    using ARRAY_T = decltype(hn::Undefined(d));

    vectorXoshiroPlusPlus<ARRAY_T> gernerator{seed};
    const auto state = gernerator.getState();
    XoshiroPlusPlus reference{seed};
    const auto lanes = hn::Lanes(d);
    auto index = 0;
    for (auto i = 0UL; i <lanes; ++i) {
        for (auto j = 0UL; j < XoshiroPlusPlus::stateSize(); ++j) {
            INFO("i = " << i);
            INFO("j = " << j);
            REQUIRE(state[index++] == reference.getState()[j]);
        }
        reference.jump();
    }
}

void test_rng(const std::uint64_t seed) {
    const auto result_array = hwy::MakeUniqueAlignedArray<std::uint64_t>(tests);
    rngLoop(seed, result_array.get(), tests);
    XoshiroPlusPlus reference{seed};
    const hn::ScalableTag<std::uint64_t> d;
    const auto lanes = hn::Lanes(d);

    for (auto i = 0UL; i < tests; i+=lanes) {
        REQUIRE(result_array[i] == reference());
    }
}

void test_uniform(const std::uint64_t seed) {
    const auto result_array = hwy::MakeUniqueAlignedArray<double>(tests);
    uniformLoop(seed, result_array.get(), tests);
    XoshiroPlusPlus reference{seed};
    const hn::ScalableTag<double> d;
    const auto lanes = hn::Lanes(d);
    for (auto i = 0UL; i < tests; i+=lanes) {
        REQUIRE(result_array[i] == reference.uniform());
    }
}

}  // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();  // required if not using HWY_ATTR

#if HWY_ONCE

// This macro declares a static array used for dynamic dispatch.
HWY_EXPORT(rngLoop);
HWY_EXPORT(test_seeding);
HWY_EXPORT(test_rng);
HWY_EXPORT(test_uniform);

void CallMulRngLoop(std::uint64_t seed, std::uint64_t* HWY_RESTRICT result, const size_t size) {
    // This must reside outside of HWY_NAMESPACE because it references (calls
    // the appropriate one from) the per-target implementations there. For
    // static dispatch, use HWY_STATIC_DISPATCH.
    return HWY_DYNAMIC_DISPATCH(rngLoop)(seed, result, size);
}


void call_test_seeding(const std::uint64_t seed){
    return HWY_DYNAMIC_DISPATCH(test_seeding)(seed);
}

void call_test_rng(const std::uint64_t seed){
    return HWY_DYNAMIC_DISPATCH(test_rng)(seed);
}


void call_test_uniform(const std::uint64_t seed){
    return HWY_DYNAMIC_DISPATCH(test_uniform)(seed);
}
TEST_CASE("tesSeeding"){
    const auto seed = Catch::rng()();
    INFO("seed = " << seed);
    call_test_seeding(seed);
}


TEST_CASE("testRNG"){
    const auto seed = Catch::rng()();
    INFO("seed = " << seed);
    call_test_rng(seed);
}

TEST_CASE("testUniform"){
    const auto seed = Catch::rng()();
    INFO("seed = " << seed);
    call_test_uniform(seed);
}

#endif
