#include <iostream>
#include <nanobench.h>
#include <random>
#include <xoshiro/vectorXoshiro.hpp>


static constexpr auto iterations = 1;
int main() {
  const auto seed = 42;
  std::cout << "SEED: " << seed << std::endl;
  xoshiro::VectorXoshiroNative rng(seed);
  xoshiro::Xoshiro reference(seed);
  xoshiro::VectorXoshiro dispatch(seed);
  std::uniform_real_distribution double_dist(0.0, 1.0);
  std::mt19937_64 mt(seed);
  using ankerl::nanobench::doNotOptimizeAway;
  ankerl::nanobench::Bench().minEpochIterations(16777)
    .run("Vector Xorshiro UINT64", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway( rng());
  }).run("Scalar Xorshiro UINT64", [&] {
       for (int i = 0; i < iterations; ++i) doNotOptimizeAway(reference());
  }).run("Dispatch Xorshiro UINT64", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(dispatch());
  }).run("MersenneTwister UINT64", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(mt());
  }).run("Vector Xorshiro DOUBLE", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(rng.uniform());
  }).run("Scalar Xorshiro DOUBLE", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(reference.uniform());
  }).run("Dispatch Xorshiro DOUBLE", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(dispatch.uniform());
  }).run("Vector Xorshiro std::random<double>", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(double_dist(rng));
  }).run("Scalar Xorshiro std::random<double>", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(double_dist(reference));
  }).run("MersenneTwister std::random<double>", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(double_dist(mt));
  });
  return 0;
}