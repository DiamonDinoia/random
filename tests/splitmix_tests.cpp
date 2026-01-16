#include <cstdint>
#include <random>
#include <catch2/catch_all.hpp>
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
