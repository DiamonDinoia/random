//
// Created by mbarbone on 5/9/23.
//

#include <VectorXoshiro/xoshiroPlusPlus.h>

#include <xoshiro256plusplus.c>
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

static constexpr auto tests = 1 << 15;

TEST_CASE("xoshiro256plusplus", "[xoshiro256plusplus]") {
    const auto seed = std::random_device()();
    INFO("SEED: " << seed);
    XoshiroPlusPlus xoshiroPlusPlus(seed);
    s[0] = xoshiroPlusPlus.getState()[0];
    s[1] = xoshiroPlusPlus.getState()[1];
    s[2] = xoshiroPlusPlus.getState()[2];
    s[3] = xoshiroPlusPlus.getState()[3];
    for (int i = 0; i < tests; ++i) { REQUIRE(xoshiroPlusPlus() == next()); }
    xoshiroPlusPlus.jump();
    jump();
    REQUIRE(xoshiroPlusPlus.getState()[0] == s[0]);
    REQUIRE(xoshiroPlusPlus.getState()[1] == s[1]);
    REQUIRE(xoshiroPlusPlus.getState()[2] == s[2]);
    REQUIRE(xoshiroPlusPlus.getState()[3] == s[3]);
    for (int i = 0; i < tests; ++i) {
        const auto result = xoshiroPlusPlus.uniform();
        REQUIRE(result >= 0);
        REQUIRE(result < 1);
    }
}

