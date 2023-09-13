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

HWY_BEFORE_NAMESPACE();  // required if not using HWY_ATTR

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
        //
        auto index = 0;
        for (auto i = 0UL; i < lanes; ++i) {
            for (auto j = 0UL; j < XoshiroPlusPlus::stateSize(); ++j) { state[index++] = stateArray[lanes * j + i]; }
        }
        return state;
    }

    constexpr auto uniform() noexcept {
        namespace hn    = hwy::HWY_NAMESPACE;
        const auto bits = hn::ShiftRight<11>(next());
        const auto real = hn::ConvertTo(fl, bits);
        return real * MUL_VALUE;
    }

    static constexpr std::uint64_t min() noexcept { return std::numeric_limits<std::uint64_t>::min(); }
    static constexpr std::uint64_t max() noexcept { return std::numeric_limits<std::uint64_t>::max(); }

   private:
    T m_state[4];
    //
    static constexpr hwy::HWY_NAMESPACE::ScalableTag<double> fl;
    //
    const decltype(hwy::HWY_NAMESPACE::Undefined(fl)) MUL_VALUE = hwy::HWY_NAMESPACE::Set(fl, 0x1.0p-53);

    //
    template <int k>
    static inline constexpr T rotl(const T x) noexcept {
        namespace hn = hwy::HWY_NAMESPACE;
        return hn::Or(hn::ShiftLeft<k>(x), hn::ShiftRight<64 - k>(x));
    }

    T next() noexcept {
        namespace hn   = hwy::HWY_NAMESPACE;
        const T result = hn::Add(rotl<23>(hn::Add(m_state[0], m_state[3])), m_state[0]);
        const T t      = hn::ShiftLeft<17>(m_state[1]);
        //
        m_state[2] = hn::Xor(m_state[2], m_state[0]);
        m_state[3] = hn::Xor(m_state[3], m_state[1]);
        m_state[1] = hn::Xor(m_state[1], m_state[2]);
        m_state[0] = hn::Xor(m_state[0], m_state[3]);
        m_state[2] = hn::Xor(m_state[2], t);
        m_state[3] = rotl<45>(m_state[3]);
        return result;
    }
};

}  // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();  // required if not using HWY_ATTR
