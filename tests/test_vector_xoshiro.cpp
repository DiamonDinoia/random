#include <xoshiro/vectorXoshiro.h>

#include <xoshiro256plusplus.c>
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

static constexpr auto tests = 1 << 12; // 4096

TEST_CASE("SEED", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  xoshiro::Xoshiro reference(seed);
  xoshiro::VectorXoshiro rng(seed);
  REQUIRE(rng.getState(0) == reference.getState());
  for (auto i = 1UL; i < xoshiro::VectorXoshiro::SIMD_WIDTH; ++i) {
      reference.jump();
      REQUIRE(rng.getState(i) == reference.getState());
  }
}

TEST_CASE("JUMP", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  xoshiro::Xoshiro reference(seed);
  xoshiro::VectorXoshiro rng(seed);
  for (auto i = 1U; i < xoshiro::VectorXoshiro::SIMD_WIDTH; ++i) {
    reference.jump();
  }
  rng.jump();
  for (auto i = 0U; i < xoshiro::VectorXoshiro::SIMD_WIDTH; ++i) {
    INFO( "i: " << i);
    REQUIRE(rng.getState(i) == reference.getState());
    reference.jump();
  }
}

TEST_CASE("LONG JUMP", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  xoshiro::Xoshiro reference(seed);
  xoshiro::VectorXoshiro rng(seed);
  rng.long_jump();
  reference.long_jump();
  for (auto i = 0UL; i < xoshiro::VectorXoshiro::SIMD_WIDTH; ++i) {
    INFO( "i: " << i);
    REQUIRE(rng.getState(i) == reference.getState());
    reference.jump();
  }
}

TEST_CASE("GENERATE UINT64", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  xoshiro::VectorXoshiro rng(seed);
  std::vector<xoshiro::Xoshiro> reference;
  reference.reserve(xoshiro::VectorXoshiro::SIMD_WIDTH);
  for (auto i = 0UL; i < xoshiro::VectorXoshiro::SIMD_WIDTH; ++i) {
    reference.emplace_back(seed);
  }
  for (auto i = 1U; i < xoshiro::VectorXoshiro::SIMD_WIDTH; ++i) {
    for (auto j = 0UL; j < i; ++j) {
      reference[i].jump();
    }
  }
  for (auto i = 0; i < xoshiro::VectorXoshiro::SIMD_WIDTH; ++i) {
    REQUIRE(rng.getState(i) == reference[i].getState());
  }
  for (auto i = 0; i < tests; i+=xoshiro::VectorXoshiro::SIMD_WIDTH) {
    for (auto j = 0; j < xoshiro::VectorXoshiro::SIMD_WIDTH; ++j) {
      INFO("i: " << i << " j: " << j);
      REQUIRE(rng() == reference[j]());
    }
  }
}

TEST_CASE("GENERATE DOUBLE", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  xoshiro::VectorXoshiro rng(seed);
  std::vector<xoshiro::Xoshiro> reference;
  reference.reserve(xoshiro::VectorXoshiro::SIMD_WIDTH);
  for (auto i = 0UL; i < xoshiro::VectorXoshiro::SIMD_WIDTH; ++i) {
    reference.emplace_back(seed);
  }
  for (auto i = 1U; i < xoshiro::VectorXoshiro::SIMD_WIDTH; ++i) {
    for (auto j = 0UL; j < i; ++j) {
      reference[i].jump();
    }
  }
  for (auto i = 0; i < xoshiro::VectorXoshiro::SIMD_WIDTH; ++i) {
    REQUIRE(rng.getState(i) == reference[i].getState());
  }
  for (auto i = 0; i < tests; i+=xoshiro::VectorXoshiro::SIMD_WIDTH) {
    for (auto j = 0; j < xoshiro::VectorXoshiro::SIMD_WIDTH; ++j) {
      INFO("i: " << i << " j: " << j);
      REQUIRE(rng.uniform() == reference[j].uniform());
    }
  }
}