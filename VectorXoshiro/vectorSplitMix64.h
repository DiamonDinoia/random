//
// Created by mbarbone on 5/9/23.
//

//#ifndef CPP_LEARNING_VECTORSPLITMIX64_H
//#define CPP_LEARNING_VECTORSPLITMIX64_H

#include <cstdint>
#include <limits>
#include <hwy/foreach_target.h>                        // must come before highway.h included by splitmix
#include <hwy/highway.h>


HWY_BEFORE_NAMESPACE();    // required if not using HWY_ATTR

namespace HWY_NAMESPACE {  // required: unique per target

template <typename T>
class VectorSplitMix64 {
   public:
    inline constexpr VectorSplitMix64(const T state) noexcept : m_state(state) {}
    inline constexpr VectorSplitMix64(const VectorSplitMix64& other) noexcept            = default;
    inline constexpr VectorSplitMix64(VectorSplitMix64&& other) noexcept                 = default;
    inline constexpr VectorSplitMix64& operator=(const VectorSplitMix64& other) noexcept = default;
    inline constexpr VectorSplitMix64& operator=(VectorSplitMix64&& other) noexcept      = default;
    //
    inline constexpr auto operator()() {
        namespace hn = hwy::HWY_NAMESPACE;
        //
        auto z = m_state +  hn::Set(hn::DFromV<T>(), 0x9e3779b97f4a7c15);
        z = hn::Mul(hn::Xor(z, hn::ShiftRight<30>(z)), hn::Set(hn::DFromV<T>(), 0xbf58476d1ce4e5b9));
        z = hn::Mul(hn::Xor(z, hn::ShiftRight<27>(z)), hn::Set(hn::DFromV<T>(), 0x94d049bb133111eb));
        return hn::Xor(z, hn::ShiftRight<31>(z));
    }
    //
    constexpr const T& getState() const { return m_state; }
    constexpr void     setState(const T& state) { m_state = state; }

    T m_state;
};

}  // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();  // required if not using HWY_ATTR

//#endif  // CPP_LEARNING_VECTORSPLITMIX64_H
