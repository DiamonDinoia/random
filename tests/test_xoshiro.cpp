#include <cstdint>
#include <random>
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>
#include <random/xoshiro_scalar.hpp>

namespace {
uint64_t s[4];

inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t reference_next() {
    const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

void reference_jump() {
    static const uint64_t JUMP[] = {0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
                                    0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL};
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 64; b++) {
            if (JUMP[i] & (1ULL << b)) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            reference_next();
        }
    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}

void reference_long_jump() {
    static const uint64_t LONG_JUMP[] = {0x76e15d3efefdcbbfULL, 0xc5004e441c522fb3ULL,
                                         0x77710069854ee241ULL, 0x39109bb02acbe635ULL};
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 64; b++) {
            if (LONG_JUMP[i] & (1ULL << b)) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            reference_next();
        }
    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}
}

static constexpr auto tests = 1 << 15;

TEST_CASE("xoshiro256++", "[xoshiro256++]") {
    const auto seed = std::random_device()();
    INFO("SEED: " << seed);
    prng::XoshiroScalar rng(seed);
    s[0] = rng.getState()[0];
    s[1] = rng.getState()[1];
    s[2] = rng.getState()[2];
    s[3] = rng.getState()[3];
    for (int i = 0; i < tests; ++i) { REQUIRE(rng() == reference_next()); }
    rng.jump();
    reference_jump();
    REQUIRE(rng.getState()[0] == s[0]);
    REQUIRE(rng.getState()[1] == s[1]);
    REQUIRE(rng.getState()[2] == s[2]);
    REQUIRE(rng.getState()[3] == s[3]);
    for (int i = 0; i < tests; ++i) {
        const auto result = rng.uniform();
        REQUIRE(result >= 0);
        REQUIRE(result < 1);
    }
}
