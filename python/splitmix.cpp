#include <nanobind/nanobind.h>
#include "xoshiro/splitMix64.hpp"

namespace nb = nanobind;
using namespace xoshiro;

class PySplitMix64 {
public:
  // Construct the generator with a seed
  PySplitMix64(uint64_t seed) : gen(seed) {}

  // Generate the next 64-bit random number
  uint64_t random_raw() {
    return gen();
  }

  // Get the current state
  uint64_t get_state() const {
    return gen.getState();
  }

  // Set the internal state
  void set_state(uint64_t state) {
    gen.setState(state);
  }

private:
  SplitMix64 gen;
};

NB_MODULE(pyrandom, m) {
  nb::class_<PySplitMix64>(m, "SplitMix64")
      .def(nb::init<uint64_t>())
      .def("random_raw", &PySplitMix64::random_raw)
      .def("get_state", &PySplitMix64::get_state)
      .def("set_state", &PySplitMix64::set_state);
}
