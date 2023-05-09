//
// Created by mbarbone on 5/9/23.
//

#include <VectorXoroshiro/splitMix64.h>
#include <splitmix64.c>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>


TEST_CASE("splitmix64", "[splitmix64]") {
    x = 0xdeadbeef;
    SplitMix64 splitMix64(0xdeadbeef);
    for (int i = 0; i < 100; ++i) {
        REQUIRE(splitMix64() == next());
    }
}




