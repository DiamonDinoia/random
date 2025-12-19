#include <iostream>
#include <nanobench.h>
#include <random>
#include <random/xoshiro_simd.hpp>

#include "xoshiro256plusplus.c"

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
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(next());
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
