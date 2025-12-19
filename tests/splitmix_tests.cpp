#include <cstdint>
#include <random>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <random/splitmix.hpp>

#include "splitmix64.c"

static constexpr auto tests = 1 << 15;

TEST_CASE("splitmix", "[splitmix]") {
    x = std::random_device{}();
    INFO("SEED: " << x);
    prng::SplitMix splitmix(x);
    for (int i = 0; i < tests; ++i) {
        REQUIRE(splitmix() == next());
    }
}

