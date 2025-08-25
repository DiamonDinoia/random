#include <nanobind/nanobind.h>
#include "random/splitmix.hpp"
#include "random/xoshiro.hpp"
#include "random/xoshiro_simd.hpp"
#include <numpy/random/bitgen.h>

namespace nb = nanobind;
using namespace prng;

class PySplitMix {
public:
  PySplitMix(uint64_t seed) : gen(seed) {}
  uint64_t random_raw() { return gen(); }
  uint64_t get_state() const { return gen.getState(); }
  void set_state(uint64_t state) { gen.setState(state); }

private:
  SplitMix gen;
};

struct SplitMixBitGen {
  bitgen_t table;
  SplitMix rng;
  explicit SplitMixBitGen(uint64_t seed) : table{}, rng(seed) {}
};

static uint64_t splitmix_uint64(void *state) {
  return static_cast<SplitMixBitGen *>(state)->rng();
}

static uint32_t splitmix_uint32(void *state) {
  return static_cast<uint32_t>(splitmix_uint64(state) >> 32);
}

static double splitmix_double(void *state) {
  return static_cast<double>(splitmix_uint64(state) >> 11) * 0x1.0p-53;
}

static uint64_t splitmix_raw(void *state) { return splitmix_uint64(state); }

static void splitmix_bitgen_capsule_free(PyObject *capsule) {
  auto *p = static_cast<SplitMixBitGen *>(PyCapsule_GetPointer(capsule, "BitGenerator"));
  delete p;
}

static nb::object make_splitmix_bitgenerator(uint64_t seed) {
  auto *payload = new SplitMixBitGen(seed);
  payload->table.state = payload;
  payload->table.next_uint64 = splitmix_uint64;
  payload->table.next_uint32 = splitmix_uint32;
  payload->table.next_double = splitmix_double;
  payload->table.next_raw = splitmix_raw;

  PyObject *capsule = PyCapsule_New(&payload->table, "BitGenerator", splitmix_bitgen_capsule_free);
  return nb::steal(capsule);
}

// Wrapper around the Xoshiro generator
class PyXoshiroNative {
public:
  PyXoshiroNative(uint64_t seed) : rng(seed) {}
  PyXoshiroNative(uint64_t seed, uint64_t thread) : rng(seed, thread) {}
  PyXoshiroNative(uint64_t seed, uint64_t thread, uint64_t cluster) : rng(seed, thread, cluster) {}

  uint64_t random_raw() { return rng(); }
  double uniform() { return rng.uniform(); }
  std::array<uint64_t, 4> get_state() const { return rng.getState(0); }
  void jump() { rng.jump(); }
  void long_jump() { rng.long_jump(); }

private:
  XoshiroNative rng;
};

struct XoshiroBitGen {
  bitgen_t table;
  Xoshiro rng;
  explicit XoshiroBitGen(uint64_t seed) : table{}, rng(seed) {}
};

static uint64_t xoshiro_uint64(void *state) {
  return static_cast<XoshiroBitGen *>(state)->rng();
}

static uint32_t xoshiro_uint32(void *state) {
  return static_cast<uint32_t>(xoshiro_uint64(state) >> 32);
}

static double xoshiro_double(void *state) {
  return static_cast<XoshiroBitGen *>(state)->rng.uniform();
}

static uint64_t xoshiro_raw(void *state) { return xoshiro_uint64(state); }

static void xoshiro_bitgen_capsule_free(PyObject *capsule) {
  auto *p = static_cast<XoshiroBitGen *>(PyCapsule_GetPointer(capsule, "BitGenerator"));
  delete p;
}


static nb::object make_xoshiro_bitgenerator(uint64_t seed) {
  auto *payload = new XoshiroBitGen(seed);
  payload->table.state = payload;
  payload->table.next_uint64 = xoshiro_uint64;
  payload->table.next_uint32 = xoshiro_uint32;
  payload->table.next_double = xoshiro_double;
  payload->table.next_raw = xoshiro_raw;

  PyObject *capsule = PyCapsule_New(&payload->table, "BitGenerator", xoshiro_bitgen_capsule_free);
  return nb::steal(capsule);
}

class PyXoshiroSIMD {
public:
  PyXoshiroSIMD(uint64_t seed) : rng(seed) {}
  PyXoshiroSIMD(uint64_t seed, uint64_t thread) : rng(seed, thread) {}
  PyXoshiroSIMD(uint64_t seed, uint64_t thread, uint64_t cluster) : rng(seed, thread, cluster) {}

  uint64_t random_raw() { return rng(); }
  double uniform() { return rng.uniform(); }
  void jump() { rng.jump(); }
  void long_jump() { rng.long_jump(); }

private:
  XoshiroSIMD rng;
};


NB_MODULE(pyrandom_ext, m) {
  nb::class_<PySplitMix>(m, "SplitMix")
      .def(nb::init<uint64_t>())
      .def("random_raw", &PySplitMix::random_raw)
      .def("get_state", &PySplitMix::get_state)
      .def("set_state", &PySplitMix::set_state);

  nb::class_<PyXoshiroNative>(m, "XoshiroNative")
      .def(nb::init<uint64_t>())
      .def(nb::init<uint64_t, uint64_t>())
      .def(nb::init<uint64_t, uint64_t, uint64_t>())
      .def("random_raw", &PyXoshiroNative::random_raw)
      .def("uniform", &PyXoshiroNative::uniform)
      .def("get_state", &PyXoshiroNative::get_state)
      .def("jump", &PyXoshiroNative::jump)
      .def("long_jump", &PyXoshiroNative::long_jump);

  nb::class_<PyXoshiroSIMD>(m, "XoshiroSIMD")
      .def(nb::init<uint64_t>())
      .def(nb::init<uint64_t, uint64_t>())
      .def(nb::init<uint64_t, uint64_t, uint64_t>())
      .def("random_raw", &PyXoshiroSIMD::random_raw)
      .def("uniform", &PyXoshiroSIMD::uniform)
      .def("jump", &PyXoshiroSIMD::jump)
      .def("long_jump", &PyXoshiroSIMD::long_jump);

  m.def("create_bit_generator", &make_xoshiro_bitgenerator, nb::arg("seed"),
        "Return a NumPy BitGenerator backed by XoshiroSIMD");
  m.def("create_splitmix_bit_generator", &make_splitmix_bitgenerator, nb::arg("seed"),
        "Return a NumPy BitGenerator backed by SplitMix");
}
