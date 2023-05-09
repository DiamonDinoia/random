//
// Created by mbarbone on 5/9/23.
//

#ifndef VECTORXOROSHIRO_SPLITMIX64_H
#define VECTORXOROSHIRO_SPLITMIX64_H

#include <cstdint>
#include <limits>

class SplitMix64 {
   public:
    inline constexpr SplitMix64(const std::uint64_t state) noexcept : m_state(state) {}
    inline constexpr SplitMix64(const SplitMix64& other) noexcept              = default;
    inline constexpr SplitMix64(SplitMix64&& other) noexcept                   = default;
    inline constexpr SplitMix64&   operator=(const SplitMix64& other) noexcept = default;
    inline constexpr SplitMix64&   operator=(SplitMix64&& other) noexcept      = default;

    inline constexpr std::uint64_t operator()() {
        std::uint64_t z = (m_state += 0x9e3779b97f4a7c15);
        z               = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z               = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }

    inline constexpr std::uint64_t min() noexcept { return std::numeric_limits<std::uint64_t>::lowest(); }

    inline constexpr std::uint64_t max() noexcept { return std::numeric_limits<std::uint64_t>::max(); }

    constexpr std::uint64_t        getState() const { return m_state; }

    constexpr void                 setState(std::uint64_t state) { m_state = state; }

   private:
    std::uint64_t m_state;
};

#endif  // VECTORXOROSHIRO_SPLITMIX64_H
