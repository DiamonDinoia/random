//
// Created by mbarbone on 5/9/23.
//

#include <VectorXoshiro/splitMix64.h>
#include <splitmix64.c>
#include <random>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

static constexpr auto tests  = 1<<15;

TEST_CASE("splitmix64", "[splitmix64]") {
    x = std::random_device()();
    INFO("SEED: " << x);
    SplitMix64 splitMix64(x);
    for (int i = 0; i < tests; ++i) {
        REQUIRE(splitMix64() == next());
    }
}




