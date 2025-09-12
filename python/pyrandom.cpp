#include <nanobind/nanobind.h>
#include <numpy/random/bitgen.h>
#include <type_traits>  // std::void_t, std::true_type, std::false_type
#include <utility>      // std::forward

#include "random/macros.hpp"
#include "random/splitmix.hpp"
#include "random/xoshiro.hpp"
#include "random/xoshiro_simd.hpp"

namespace nb = nanobind;
using namespace prng;

// -----------------------------------------------------------------------------
// Python-exposed wrappers (unchanged API)
class PySplitMix {
public:
  PRNG_ALWAYS_INLINE explicit PySplitMix(uint64_t seed) noexcept : gen(seed) {}
  PRNG_ALWAYS_INLINE uint64_t random_raw() noexcept { return gen(); }
  PRNG_ALWAYS_INLINE uint64_t get_state() const noexcept { return gen.getState(); }
  PRNG_ALWAYS_INLINE void set_state(uint64_t s) noexcept { gen.setState(s); }

private:
  SplitMix gen;
};

class PyXoshiroNative {
public:
  PRNG_ALWAYS_INLINE explicit PyXoshiroNative(uint64_t seed) noexcept : rng(seed) {}
  PRNG_ALWAYS_INLINE PyXoshiroNative(uint64_t seed, uint64_t thread) noexcept : rng(seed, thread) {}
  PRNG_ALWAYS_INLINE PyXoshiroNative(uint64_t seed, uint64_t thread, uint64_t cluster) noexcept
      : rng(seed, thread, cluster) {}

  PRNG_ALWAYS_INLINE uint64_t random_raw() noexcept { return rng(); }
  PRNG_ALWAYS_INLINE double uniform() noexcept { return rng.uniform(); }
  PRNG_ALWAYS_INLINE void jump() noexcept { rng.jump(); }
  PRNG_ALWAYS_INLINE void long_jump() noexcept { rng.long_jump(); }

private:
  XoshiroNative rng;
};

class PyXoshiroSIMD {
public:
  PRNG_ALWAYS_INLINE explicit PyXoshiroSIMD(uint64_t seed) noexcept : rng(seed) {}
  PRNG_ALWAYS_INLINE PyXoshiroSIMD(uint64_t seed, uint64_t thread) noexcept : rng(seed, thread) {}
  PRNG_ALWAYS_INLINE PyXoshiroSIMD(uint64_t seed, uint64_t thread, uint64_t cluster) noexcept
      : rng(seed, thread, cluster) {}

  PRNG_ALWAYS_INLINE uint64_t random_raw() noexcept { return rng(); }
  PRNG_ALWAYS_INLINE double uniform() noexcept { return rng.uniform(); }
  PRNG_ALWAYS_INLINE void jump() noexcept { rng.jump(); }
  PRNG_ALWAYS_INLINE void long_jump() noexcept { rng.long_jump(); }

private:
  XoshiroSIMD rng;
};

// -----------------------------------------------------------------------------
// Traits: detect presence of rng.uniform()
template <typename T, typename = void> struct has_uniform : std::false_type {};
template <typename T>
struct has_uniform<T, std::void_t<decltype(std::declval<T&>().uniform())>> : std::true_type {};

// -----------------------------------------------------------------------------
// DirectBitGen: optimized adapter
template <typename Rng>
struct alignas(64) DirectBitGen {
  Rng      rng;
  bitgen_t base;

  template <typename... Args>
  PRNG_ALWAYS_INLINE explicit DirectBitGen(Args&&... args) noexcept
  : rng(std::forward<Args>(args)...), base{} {
    base.state       = this;
    base.next_uint64 = &DirectBitGen::next_u64;
    base.next_uint32 = &DirectBitGen::next_u32;
    base.next_double = &DirectBitGen::next_f64;
    base.next_raw    = base.next_uint64;
  }

  // Static callbacks — no capturing lambdas, no thunks
  PRNG_ALWAYS_INLINE static uint64_t next_u64(void* s) noexcept {
    auto* self = static_cast<DirectBitGen*>(s);
    return self->rng(); // rely on inlining in Rng::operator()
  }

  PRNG_ALWAYS_INLINE static uint32_t next_u32(void* s) noexcept {
    auto* self = static_cast<DirectBitGen*>(s);
    // If your RNG has a native 32-bit fast path, you can specialize it here.
    return static_cast<uint32_t>(self->rng() >> 32);
  }

  PRNG_ALWAYS_INLINE static double next_f64(void* s) noexcept {
    auto* self = static_cast<DirectBitGen*>(s);
    if constexpr (has_uniform<Rng>::value) {
      return self->rng.uniform();
    } else {
      // 53-bit mantissa from upper bits
      return static_cast<double>(self->rng() >> 11) * 0x1.0p-53;
    }
  }
};

// -----------------------------------------------------------------------------
// Capsule management: keep NumPy-facing pointer = bitgen_t*, but own the
// DirectBitGen<T> via the capsule *context*, so we can delete safely.
template <typename Generator>
static void capsule_destruct(PyObject* capsule) noexcept {
  // Retrieve the owner pointer from the context (not the capsule pointer)
  void* ctx = PyCapsule_GetContext(capsule);
  auto* gen = static_cast<Generator*>(ctx);
  delete gen;
}

template <typename Generator>
static nb::object make_direct_bitgenerator_capsule(Generator* gen) {
  // NumPy expects a capsule with a "BitGenerator" pointer to bitgen_t
  bitgen_t* base_ptr = &gen->base;
  PyObject* cap = PyCapsule_New(static_cast<void*>(base_ptr), "BitGenerator",
                                &capsule_destruct<Generator>);
  // Store the actual owner so the destructor can delete it
  PyCapsule_SetContext(cap, static_cast<void*>(gen));
  return nb::steal(cap);
}

// -----------------------------------------------------------------------------
// Factory helpers
template <typename Rng, typename... Args>
PRNG_ALWAYS_INLINE nb::object make_direct_bitgenerator(Args&&... args) {
  using Generator = DirectBitGen<Rng>;
  auto* gen = new Generator(std::forward<Args>(args)...);
  return make_direct_bitgenerator_capsule(gen);
}

PRNG_ALWAYS_INLINE nb::object make_splitmix_bitgenerator(uint64_t seed) {
  return make_direct_bitgenerator<SplitMix>(seed);
}

PRNG_ALWAYS_INLINE nb::object make_xoshiro_bitgenerator(uint64_t seed) {
  return make_direct_bitgenerator<Xoshiro>(seed);
}

PRNG_ALWAYS_INLINE nb::object make_xoshiro_simd_bitgenerator(uint64_t seed) {
  return make_direct_bitgenerator<XoshiroSIMD>(seed);
}

PRNG_ALWAYS_INLINE nb::object make_xoshiro_native_bitgenerator(uint64_t seed) {
  return make_direct_bitgenerator<XoshiroNative>(seed);
}
PRNG_ALWAYS_INLINE nb::object make_xoshiro_native_bitgenerator(uint64_t seed, uint64_t thread) {
  return make_direct_bitgenerator<XoshiroNative>(seed, thread);
}
PRNG_ALWAYS_INLINE nb::object make_xoshiro_native_bitgenerator(uint64_t seed, uint64_t thread, uint64_t cluster) {
  return make_direct_bitgenerator<XoshiroNative>(seed, thread, cluster);
}

// -----------------------------------------------------------------------------
// Python module
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

  // NumPy BitGenerator factories
  m.def("create_bit_generator", &make_xoshiro_simd_bitgenerator, nb::arg("seed"),
        "Return a NumPy BitGenerator backed by XoshiroSIMD");

  m.def("create_splitmix_bit_generator", &make_splitmix_bitgenerator, nb::arg("seed"),
        "Return a NumPy BitGenerator backed by SplitMix");

  m.def("create_xoshiro_bit_generator", &make_xoshiro_bitgenerator, nb::arg("seed"),
        "Return a NumPy BitGenerator backed by Xoshiro");

  m.def("create_xoshiro_native_bit_generator",
        nb::overload_cast<uint64_t>(&make_xoshiro_native_bitgenerator),
        nb::arg("seed"),
        "Return a NumPy BitGenerator backed by XoshiroNative (seed)");

  m.def("create_xoshiro_native_bit_generator",
        nb::overload_cast<uint64_t, uint64_t>(&make_xoshiro_native_bitgenerator),
        nb::arg("seed"), nb::arg("thread"),
        "Return a NumPy BitGenerator backed by XoshiroNative (seed, thread)");

  m.def("create_xoshiro_native_bit_generator",
        nb::overload_cast<uint64_t, uint64_t, uint64_t>(&make_xoshiro_native_bitgenerator),
        nb::arg("seed"), nb::arg("thread"), nb::arg("cluster"),
        "Return a NumPy BitGenerator backed by XoshiroNative (seed, thread, cluster)");
}
