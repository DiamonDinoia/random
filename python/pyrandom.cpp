#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include "xoshiro/splitMix64.hpp"
#include "xoshiro/xoshiro.hpp"
#include "xoshiro/vectorXoshiro.hpp"
#include <numpy/random/bitgen.h>

namespace nb = nanobind;
using namespace xoshiro;

class PySplitMix64 {
public:
  PySplitMix64(uint64_t seed) : gen(seed) {}
  uint64_t random_raw() { return gen(); }
  uint64_t get_state() const { return gen.getState(); }
  void set_state(uint64_t state) { gen.setState(state); }

private:
  SplitMix64 gen;
};

// Wrapper around the Xoshiro generator
class PyXoshiro {
public:
  PyXoshiro(uint64_t seed) : rng(seed) {}
  PyXoshiro(uint64_t seed, uint64_t thread) : rng(seed, thread) {}
  PyXoshiro(uint64_t seed, uint64_t thread, uint64_t cluster) : rng(seed, thread, cluster) {}

  uint64_t random_raw() { return rng(); }
  double uniform() { return rng.uniform(); }
  std::array<uint64_t, 4> get_state() const { return rng.getState(); }
  void jump() { rng.jump(); }
  void long_jump() { rng.long_jump(); }

private:
  Xoshiro rng;
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

class PyVectorXoshiro {
public:
  PyVectorXoshiro(uint64_t seed) : rng(seed) {}
  PyVectorXoshiro(uint64_t seed, uint64_t thread) : rng(seed, thread) {}
  PyVectorXoshiro(uint64_t seed, uint64_t thread, uint64_t cluster) : rng(seed, thread, cluster) {}

  uint64_t random_raw() { return rng(); }
  double uniform() { return rng.uniform(); }
  void jump() { rng.jump(); }
  void long_jump() { rng.long_jump(); }

private:
  VectorXoshiro rng;
};

class PyVectorXoshiroNative {
public:
  PyVectorXoshiroNative(uint64_t seed) : rng(seed) {}
  PyVectorXoshiroNative(uint64_t seed, uint64_t thread) : rng(seed, thread) {}
  PyVectorXoshiroNative(uint64_t seed, uint64_t thread, uint64_t cluster) : rng(seed, thread, cluster) {}

  uint64_t random_raw() { return rng(); }
  double uniform() { return rng.uniform(); }
  void jump() { rng.jump(); }
  void long_jump() { rng.long_jump(); }

private:
  VectorXoshiroNative rng;
};

struct VectorXoshiroBitGen {
  bitgen_t table;
  VectorXoshiro rng;
  explicit VectorXoshiroBitGen(uint64_t seed) : table{}, rng(seed) {}
};

static uint64_t vector_xoshiro_uint64(void *state) {
  return static_cast<VectorXoshiroBitGen *>(state)->rng();
}

static uint32_t vector_xoshiro_uint32(void *state) {
  return static_cast<uint32_t>(vector_xoshiro_uint64(state) >> 32);
}

static double vector_xoshiro_double(void *state) {
  return static_cast<VectorXoshiroBitGen *>(state)->rng.uniform();
}

static uint64_t vector_xoshiro_raw(void *state) { return vector_xoshiro_uint64(state); }

static void vector_bitgen_capsule_free(PyObject *capsule) {
  auto *p = static_cast<VectorXoshiroBitGen *>(PyCapsule_GetPointer(capsule, "BitGenerator"));
  delete p;
}

static nb::object make_vector_bitgenerator(uint64_t seed) {
  auto *payload = new VectorXoshiroBitGen(seed);
  payload->table.state = payload;
  payload->table.next_uint64 = vector_xoshiro_uint64;
  payload->table.next_uint32 = vector_xoshiro_uint32;
  payload->table.next_double = vector_xoshiro_double;
  payload->table.next_raw = vector_xoshiro_raw;

  PyObject *capsule = PyCapsule_New(&payload->table, "BitGenerator", vector_bitgen_capsule_free);
  return nb::steal(capsule);
}

NB_MODULE(pyrandom_ext, m) {
  nb::class_<PySplitMix64>(m, "SplitMix64")
      .def(nb::init<uint64_t>())
      .def("random_raw", &PySplitMix64::random_raw)
      .def("get_state", &PySplitMix64::get_state)
      .def("set_state", &PySplitMix64::set_state);

  nb::class_<PyXoshiro>(m, "Xoshiro")
      .def(nb::init<uint64_t>())
      .def(nb::init<uint64_t, uint64_t>())
      .def(nb::init<uint64_t, uint64_t, uint64_t>())
      .def("random_raw", &PyXoshiro::random_raw)
      .def("uniform", &PyXoshiro::uniform)
      .def("get_state", &PyXoshiro::get_state)
      .def("jump", &PyXoshiro::jump)
      .def("long_jump", &PyXoshiro::long_jump);

  nb::class_<PyVectorXoshiro>(m, "VectorXoshiro")
      .def(nb::init<uint64_t>())
      .def(nb::init<uint64_t, uint64_t>())
      .def(nb::init<uint64_t, uint64_t, uint64_t>())
      .def("random_raw", &PyVectorXoshiro::random_raw)
      .def("uniform", &PyVectorXoshiro::uniform)
      .def("jump", &PyVectorXoshiro::jump)
      .def("long_jump", &PyVectorXoshiro::long_jump);

  nb::class_<PyVectorXoshiroNative>(m, "VectorXoshiroNative")
      .def(nb::init<uint64_t>())
      .def(nb::init<uint64_t, uint64_t>())
      .def(nb::init<uint64_t, uint64_t, uint64_t>())
      .def("random_raw", &PyVectorXoshiroNative::random_raw)
      .def("uniform", &PyVectorXoshiroNative::uniform)
      .def("jump", &PyVectorXoshiroNative::jump)
      .def("long_jump", &PyVectorXoshiroNative::long_jump);

  m.def("create_xoshiro_bit_generator", &make_xoshiro_bitgenerator, nb::arg("seed"),
        "Return a NumPy BitGenerator backed by Xoshiro");
  m.def("create_vector_bit_generator", &make_vector_bitgenerator, nb::arg("seed"),
        "Return a NumPy BitGenerator backed by VectorXoshiro");
}
