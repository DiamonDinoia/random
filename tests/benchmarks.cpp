#include <xoshiro/vectorXoshiro.h>
#include <nanobench.h>
#include <random>

int main() {
  const auto seed = std::random_device()();
  std::cout << "SEED: " << seed << std::endl;
  xoshiro::VectorXoshiro rng(seed);
  xoshiro::Xoshiro reference(seed);
  std::uniform_real_distribution double_dist(0.0, 1.0);
  std::mt19937_64 mt(seed);
  using ankerl::nanobench::doNotOptimizeAway;
  ankerl::nanobench::Bench().minEpochIterations(10854354)
    .run("Vector Xorshiro UINT64", [&] {
    doNotOptimizeAway(rng());
  }).run("Scalar Xorshiro UINT64", [&] {
      doNotOptimizeAway(reference());
  }).run("MersenneTwister UINT64", [&] {
    doNotOptimizeAway(mt());
  }).run("Vector Xorshiro DOUBLE", [&] {
    doNotOptimizeAway(rng.uniform());
  }).run("Scalar Xorshiro DOUBLE", [&] {
    doNotOptimizeAway(reference.uniform());
  }).run("Vector Xorshiro std::random<double>", [&] {
    doNotOptimizeAway(double_dist(rng));
  }).run("Scalar Xorshiro std::random<double>", [&] {
    doNotOptimizeAway(double_dist(reference));
  }).run("MersenneTwister std::random<double>", [&] {
    doNotOptimizeAway(double_dist(mt));
  });
  return 0;
}