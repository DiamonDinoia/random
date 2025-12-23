#include <cstdint>
#include <random>
#include <cstring>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <random/chacha_scalar.hpp>
#include <iostream>

#include "chacha.c"

static constexpr auto tests = 1 << 15;

TEST_CASE("ChaChaScalar", "[chacha]") {
    using prng::ChaChaScalar;

    auto seed = std::random_device{}();
    INFO("SEED: " << seed);
    std::mt19937 rng32(seed);
    std::mt19937_64 rng64(seed);
    ChaChaScalar::input_word counter = rng64(), nonce = rng64();
    std::array<ChaChaScalar::matrix_word, 8> key;
    for (int i = 0; i < 8; i++) {
        key[i] = rng32();
    }

    prng::ChaChaScalar rngChaCha(key, counter, nonce, 20);
    for (int i = 0; i < tests; ++i) {
        // In a correct implementation, the internal state should neccsarily be
        // a validly arranged input for the algorithm and thus the reference impl.
        const auto input = rngChaCha.getState();
        u8 referenceOutput[64];
        salsa20_wordtobyte(referenceOutput, input.data());
        const auto chachaOutput = rngChaCha();
        REQUIRE(std::memcmp(referenceOutput, chachaOutput.data(), 64) == 0);
    }
}

