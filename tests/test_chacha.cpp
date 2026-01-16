#include <random>
#include <cstring>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <random/chacha.hpp>

// We test against the reference implementation but include it as a C
// library. This is because ChaCha20's reference implementation's
// constants are set like:
//      static const char sigma[16] = "expand 2-byte k"
// This definition does not fit the null terminator, which is non-compliant in
// cpp, and thus causes an -fpermissive error when included directly here.
extern "C" {
#include "chacha20_ref_impl_wrapper.h"
}

static constexpr auto tests = 1 << 15;

TEST_CASE("ChaChaScalar", "[chacha]") {
    using ChaCha20 = prng::ChaCha<20>;

    auto seed = std::random_device{}();
    INFO("SEED: " << seed);
    std::mt19937 rng32(seed);
    std::mt19937_64 rng64(seed);
    ChaCha20::input_word counter = rng64(), nonce = rng64();
    std::array<ChaCha20::matrix_word, 8> key;
    for (int i = 0; i < 8; i++) {
        key[i] = rng32();
    }

    ChaCha20 rngChaCha(key, counter, nonce);
    for (int i = 0; i < tests; ++i) {
        // In a correct implementation, the internal state should neccsarily be
        // a validly arranged input for the algorithm and thus the reference impl.
        const auto input = rngChaCha.getState();
        u8 referenceOutput[64];
        chacha20_ref_impl_wrapper(referenceOutput, input.data());
        const auto chachaOutput = rngChaCha();
        REQUIRE(std::memcmp(referenceOutput, chachaOutput.data(), 64) == 0);
    }
}

