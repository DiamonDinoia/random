//
// Created by mbarbone on 5/9/23.
//

#include <VectorXoroshiro/xoroshiroPlusPlus.h>
#include <xoshiro256plusplus.c>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

static constexpr auto tests  = 1<<15;

TEST_CASE("xoroshiro256plusplus", "[xoroshiro256plusplus]") {
    const auto seed = std::random_device()();
    INFO("SEED: " << seed);
    XoroshiroPlusPlus xoroshiroPlusPlus(seed);
    s[0] = xoroshiroPlusPlus.getState()[0];
    s[1] = xoroshiroPlusPlus.getState()[1];
    s[2] = xoroshiroPlusPlus.getState()[2];
    s[3] = xoroshiroPlusPlus.getState()[3];
    for (int i = 0; i < tests; ++i) {
        REQUIRE(xoroshiroPlusPlus() == next());
    }
}