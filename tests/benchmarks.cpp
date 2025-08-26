#include <iostream>
#include <nanobench.h>
#include <random>
#include <random/xoshiro_simd.hpp>

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
}

static constexpr auto iterations = 1;
int main() {
  using namespace std::chrono_literals;
  volatile const auto seed = 42;
  std::cout << "SEED: " << seed << std::endl;
  prng::XoshiroNative rng(seed);
  prng::XoshiroScalar reference(seed);
  prng::XoshiroSIMD dispatch(seed);

  s[0] = reference.getState()[0];
  s[1] = reference.getState()[1];
  s[2] = reference.getState()[2];
  s[3] = reference.getState()[3];

  std::uniform_real_distribution<double> double_dist(0.0, 1.0);
  std::mt19937_64 mt(seed);
  using ankerl::nanobench::doNotOptimizeAway;
  ankerl::nanobench::Bench().minEpochTime(20ms).batch(iterations)
  .run("Reference Xoshiro UINT64", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(reference_next());
  }).run("XoshiroSIMD UINT64", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(rng());
  }).run("Scalar Xoshiro UINT64", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(reference());
  }).run("Dispatch Xoshiro UINT64", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(dispatch());
  }).run("MersenneTwister UINT64", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(mt());
  }).run("XoshiroSIMD DOUBLE", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(rng.uniform());
  }).run("Scalar Xoshiro DOUBLE", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(reference.uniform());
  }).run("Dispatch Xoshiro DOUBLE", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(dispatch.uniform());
  }).run("XoshiroSIMD std::random<double>", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(double_dist(rng));
  }).run("Scalar Xoshiro std::random<double>", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(double_dist(reference));
  }).run("Dispatch Xoshiro std::random<double>", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(double_dist(dispatch));
  }).run("MersenneTwister std::random<double>", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(double_dist(mt));
  });
}
