#include <cstdint>
#include <random>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <random/splitmix.hpp>

namespace {
uint64_t state;
uint64_t reference_next() {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
}

static constexpr auto tests = 1 << 15;

TEST_CASE("splitmix", "[splitmix]") {
    state = std::random_device{}();
    INFO("SEED: " << state);
    prng::SplitMix splitmix(state);
    for (int i = 0; i < tests; ++i) {
        REQUIRE(splitmix() == reference_next());
    }
}

