//
// Created by mbarbone on 5/18/23.
//

#include <hwy/aligned_allocator.h>
#include <hwy/foreach_target.h>  // must come before highway.h included by splitmix
#include <hwy/highway.h>

#include <array>
#include <cstdint>
#include <limits>

#include "splitMix64.h"

HWY_BEFORE_NAMESPACE();    // required if not using HWY_ATTR

namespace HWY_NAMESPACE {  // required: unique per target

template <typename T>
class vectorXoroshiroPlusPlus {
   public:
    inline explicit vectorXoroshiroPlusPlus(const std::uint64_t seed) noexcept {
        namespace hn = hwy::HWY_NAMESPACE;
        SplitMix64 splitMix64{seed};
        for (auto& state : m_state) {
            auto stateArray = hwy::MakeUniqueAlignedArray<std::uint64_t>(hn::MaxLanes(hn::DFromV<T>()));
            for (std::size_t i = 0; i < hn::Lanes(hn::DFromV<T>()); ++i) { stateArray[i] = splitMix64(); }
            state = hn::Load(hn::DFromV<T>(), stateArray.get());
        }
    }

    inline constexpr vectorXoroshiroPlusPlus(const vectorXoroshiroPlusPlus& other) noexcept            = default;
    inline constexpr vectorXoroshiroPlusPlus(vectorXoroshiroPlusPlus&& other) noexcept                 = default;
    inline constexpr vectorXoroshiroPlusPlus& operator=(const vectorXoroshiroPlusPlus& other) noexcept = default;
    inline constexpr vectorXoroshiroPlusPlus& operator=(vectorXoroshiroPlusPlus&& other) noexcept      = default;
    //
    inline constexpr auto operator()() { return next(); }
    //
    static constexpr std::uint64_t stateSize() noexcept {
        namespace hn = hwy::HWY_NAMESPACE;
        return hn::Lanes(hn::DFromV<T>()) * 4;
    }

    constexpr std::array<std::uint64_t, stateSize()> getState() const {
        namespace hn = hwy::HWY_NAMESPACE;
        std::array<T, 4> state;
        for (int i = 0; i < 4; ++i) { hn::StoreU(m_state[i], hn::DFromV<T>(), &state[i]); }
        return state;
    }

    static constexpr std::uint64_t min() noexcept { return std::numeric_limits<std::uint64_t>::min(); }
    static constexpr std::uint64_t max() noexcept { return std::numeric_limits<std::uint64_t>::max(); }

   private:
    T m_state[4];
    //
    static inline constexpr T rotl(const T x, int k) noexcept {
        namespace hn = hwy::HWY_NAMESPACE;
        return hn::Or(hn::ShiftLeftSame(x, k), hn::ShiftRightSame(x, 64 - k));
    }

    T next() noexcept {
        namespace hn   = hwy::HWY_NAMESPACE;
        const T result = hn::Add(rotl(hn::Add(m_state[0], m_state[3]), 23), m_state[0]);
        const T t      = hn::ShiftLeftSame(m_state[1], 17);
        //
        m_state[2] = hn::Xor(m_state[2], m_state[0]);
        m_state[3] = hn::Xor(m_state[3], m_state[1]);
        m_state[1] = hn::Xor(m_state[1], m_state[2]);
        m_state[0] = hn::Xor(m_state[0], m_state[3]);
        m_state[2] = hn::Xor(m_state[2], t);
        m_state[3] = rotl(m_state[3], 45);
        return result;
    }
};

}  // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();  // required if not using HWY_ATTR
