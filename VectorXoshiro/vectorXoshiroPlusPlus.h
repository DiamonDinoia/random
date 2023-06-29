//
// Created by mbarbone on 5/18/23.
//

#include <hwy/aligned_allocator.h>
#include <hwy/foreach_target.h>  // must come before highway.h included by splitmix
#include <hwy/highway.h>

#include <array>
#include <cstdint>
#include <limits>

#include "xoshiroPlusPlus.h"

HWY_BEFORE_NAMESPACE();    // required if not using HWY_ATTR

namespace HWY_NAMESPACE {  // required: unique per target

template <typename T>
class vectorXoshiroPlusPlus {
   public:
    inline explicit vectorXoshiroPlusPlus(const std::uint64_t seed) {
        namespace hn = hwy::HWY_NAMESPACE;
        XoshiroPlusPlus xoshiro{seed};
        const auto      lanes      = hn::Lanes(hn::DFromV<T>());
        auto            stateArray = hwy::MakeUniqueAlignedArray<std::uint64_t>(stateSize());

        for (auto i = 0UL; i < lanes; ++i) {
            const auto state = xoshiro.getState();
            for (auto j = 0UL; j < XoshiroPlusPlus::stateSize(); ++j) {
                const auto index  = lanes * j + i;
                stateArray[index] = state[j];
            }
            xoshiro.jump();
        }

        for (auto i = 0UL; i < xoshiro.stateSize(); ++i) {
            m_state[i] = hn::Load(hn::DFromV<T>(), &stateArray[i * lanes]);
        }
    }

    inline constexpr vectorXoshiroPlusPlus(const vectorXoshiroPlusPlus& other) noexcept            = default;
    inline constexpr vectorXoshiroPlusPlus(vectorXoshiroPlusPlus&& other) noexcept                 = default;
    inline constexpr vectorXoshiroPlusPlus& operator=(const vectorXoshiroPlusPlus& other) noexcept = default;
    inline constexpr vectorXoshiroPlusPlus& operator=(vectorXoshiroPlusPlus&& other) noexcept      = default;
    //
    inline constexpr auto operator()() { return next(); }
    //
    static constexpr std::uint64_t stateSize() noexcept {
        namespace hn = hwy::HWY_NAMESPACE;
        return hn::Lanes(hn::DFromV<T>()) * XoshiroPlusPlus::stateSize();
    }

    std::array<std::uint64_t, stateSize()> getState() const {
        namespace hn          = hwy::HWY_NAMESPACE;
        const auto lanes      = hn::Lanes(hn::DFromV<T>());
        auto       stateArray = hwy::MakeUniqueAlignedArray<std::uint64_t>(stateSize());
        for (auto i = 0UL; i < XoshiroPlusPlus::stateSize(); ++i) {
            hn::Store(m_state[i], hn::DFromV<T>(), stateArray.get() + i * lanes);
        }
        std::array<std::uint64_t, stateSize()> state;
        auto                       index = 0;
        for (auto i = 0UL; i < lanes; ++i) {
            for (auto j = 0UL; j < XoshiroPlusPlus::stateSize(); ++j) { state[index++] = stateArray[lanes * j + i]; }
        }
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
