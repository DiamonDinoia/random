#include <iostream>
#include <nanobench.h>
#include <random>
#include <xoshiro/vectorXoshiro.hpp>
#include <xoshiro256plusplus.c>

static constexpr auto iterations = 1;
int main() {
  volatile const auto seed = 42;
  std::cout << "SEED: " << seed << std::endl;
  xoshiro::VectorXoshiroNative rng(seed);
  xoshiro::Xoshiro reference(seed);
  xoshiro::VectorXoshiro dispatch(seed);

  s[0] = reference.getState()[0];
  s[1] = reference.getState()[1];
  s[2] = reference.getState()[2];
  s[3] = reference.getState()[3];

  std::uniform_real_distribution double_dist(0.0, 1.0);
  std::mt19937_64 mt(seed);
  using ankerl::nanobench::doNotOptimizeAway;
  ankerl::nanobench::Bench().minEpochIterations(1 << 20)
  .run("Reference Xoshiro UINT64", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway( next());
  }).run("Vector Xoshiro UINT64", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway( rng());
  }).run("Scalar Xoshiro UINT64", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(reference());
  }).run("Dispatch Xoshiro UINT64", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(dispatch());
  }).run("MersenneTwister UINT64", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(mt());
  }).run("Vector Xoshiro DOUBLE", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(rng.uniform());
  }).run("Scalar Xoshiro DOUBLE", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(reference.uniform());
  }).run("Dispatch Xoshiro DOUBLE", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(dispatch.uniform());
  }).run("Vector Xoshiro std::random<double>", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(double_dist(rng));
  }).run("Scalar Xoshiro std::random<double>", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(double_dist(reference));
  }).run("Dispatch Xoshiro std::random<double>", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(double_dist(dispatch));
  }).run("MersenneTwister std::random<double>", [&] {
     for (int i = 0; i < iterations; ++i) doNotOptimizeAway(double_dist(mt));
  });
}
