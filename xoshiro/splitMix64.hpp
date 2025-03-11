#pragma once

#include <cstdint>
#include <limits>

namespace xoshiro {

class SplitMix64 {
public:

  constexpr explicit SplitMix64(const std::uint64_t state) noexcept : m_state(state) {}

  constexpr std::uint64_t operator()() {
    std::uint64_t z = (m_state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  }

  static constexpr std::uint64_t min() noexcept {
    return std::numeric_limits<std::uint64_t>::lowest();
  }

  static constexpr std::uint64_t max() noexcept {
    return std::numeric_limits<std::uint64_t>::max();
  }

  constexpr std::uint64_t getState() const { return m_state; }

  constexpr void setState(std::uint64_t state) { m_state = state; }

private:
  std::uint64_t m_state;
};

}