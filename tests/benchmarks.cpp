#include <iostream>
#include <nanobench.h>
#include <random>
#include <xoshiro/vectorXoshiro.hpp>

int main() {
  const auto seed = 42;
  std::cout << "SEED: " << seed << std::endl;
  xoshiro::VectorXoshiroNative rng(seed);
  xoshiro::Xoshiro reference(seed);
  xoshiro::VectorXoshiro dispatch(seed);
  std::uniform_real_distribution double_dist(0.0, 1.0);
  std::mt19937_64 mt(seed);
  using ankerl::nanobench::doNotOptimizeAway;
  ankerl::nanobench::Bench().minEpochIterations(16777216)
    .run("Vector Xorshiro UINT64", [&] {
    return doNotOptimizeAway( rng());
  }).run("Scalar Xorshiro UINT64", [&] {
      return doNotOptimizeAway(reference());
  }).run("Dispatch Xorshiro UINT64", [&] {
    return doNotOptimizeAway(dispatch());
  }).run("MersenneTwister UINT64", [&] {
    return doNotOptimizeAway(mt());
  }).run("Vector Xorshiro DOUBLE", [&] {
    return doNotOptimizeAway(rng.uniform());
  }).run("Scalar Xorshiro DOUBLE", [&] {
    return doNotOptimizeAway(reference.uniform());
  }).run("Dispatch Xorshiro DOUBLE", [&] {
    return doNotOptimizeAway(dispatch.uniform());
  }).run("Vector Xorshiro std::random<double>", [&] {
    return doNotOptimizeAway(double_dist(rng));
  }).run("Scalar Xorshiro std::random<double>", [&] {
    return doNotOptimizeAway(double_dist(reference));
  }).run("MersenneTwister std::random<double>", [&] {
    return doNotOptimizeAway(double_dist(mt));
  });
  return 0;
}