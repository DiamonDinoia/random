//
// Created by mbarbone on 5/9/23.
//

#ifndef CPP_LEARNING_XOSHIROPLUSPLUS_H
#define CPP_LEARNING_XOSHIROPLUSPLUS_H

#include <array>
#include <cstdint>
#include <limits>

#include "splitMix64.h"

class XoshiroPlusPlus {
   public:
    inline constexpr explicit XoshiroPlusPlus(const std::uint64_t seed) noexcept : m_state{} {
        SplitMix64 splitMix64{seed};
        for (auto& element : m_state) { element = splitMix64(); }
    }

    inline constexpr explicit XoshiroPlusPlus(const std::uint64_t seed, const std::uint64_t thread_id) noexcept
        : XoshiroPlusPlus(seed) {
        for (auto i = UINT64_C(0); i < thread_id; ++i) { jump(); }
    }

    inline constexpr XoshiroPlusPlus(const XoshiroPlusPlus& other) noexcept            = default;
    inline constexpr XoshiroPlusPlus(XoshiroPlusPlus&& other) noexcept                 = default;
    inline constexpr XoshiroPlusPlus& operator=(const XoshiroPlusPlus& other) noexcept = default;
    inline constexpr XoshiroPlusPlus& operator=(XoshiroPlusPlus&& other) noexcept      = default;
    //
    inline constexpr std::uint64_t operator()() noexcept { return next(); }
    //
    constexpr std::array<std::uint64_t, 4> getState() const { return {m_state[0], m_state[1], m_state[2], m_state[3]}; }
    //
    constexpr void setState(std::array<std::uint64_t, 4> state) noexcept {
        m_state[0] = state[0];
        m_state[1] = state[1];
        m_state[2] = state[2];
        m_state[3] = state[3];
    }

    static inline constexpr std::uint64_t min() noexcept { return std::numeric_limits<std::uint64_t>::min(); }
    static inline constexpr std::uint64_t max() noexcept { return std::numeric_limits<std::uint64_t>::max(); }
    static inline constexpr std::uint64_t stateSize() noexcept { return 4; }

   private:
    std::uint64_t m_state[4];
    //
    static inline constexpr std::uint64_t rotl(const std::uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }
    //
    constexpr std::uint64_t next() noexcept {
        const std::uint64_t result = rotl(m_state[0] + m_state[3], 23) + m_state[0];
        //
        const std::uint64_t t = m_state[1] << 17;

        m_state[2] ^= m_state[0];
        m_state[3] ^= m_state[1];
        m_state[1] ^= m_state[2];
        m_state[0] ^= m_state[3];

        m_state[2] ^= t;

        m_state[3] = rotl(m_state[3], 45);

        return result;
    }

   public:
    /* This is the jump function for the generator. It is equivalent
   to 2^128 calls to next(); it can be used to generate 2^128
   non-overlapping subsequences for parallel computations. */

    constexpr void jump() noexcept {
        constexpr std::uint64_t JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa,
                                          0x39abdc4529b1661c};
        //
        std::uint64_t s0 = 0;
        std::uint64_t s1 = 0;
        std::uint64_t s2 = 0;
        std::uint64_t s3 = 0;
        for (unsigned long i : JUMP)
            for (int b = 0; b < 64; b++) {
                if (i & UINT64_C(1) << b) {
                    s0 ^= m_state[0];
                    s1 ^= m_state[1];
                    s2 ^= m_state[2];
                    s3 ^= m_state[3];
                }
                next();
            }

        m_state[0] = s0;
        m_state[1] = s1;
        m_state[2] = s2;
        m_state[3] = s3;
    }

    /* This is the long-jump function for the generator. It is equivalent to
   2^192 calls to next(); it can be used to generate 2^64 starting points,
   from each of which jump() will generate 2^64 non-overlapping
   subsequences for parallel distributed computations. */

    constexpr void long_jump() noexcept {
        constexpr std::uint64_t LONG_JUMP[] = {0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241,
                                               0x39109bb02acbe635};
        std::uint64_t           s0          = 0;
        std::uint64_t           s1          = 0;
        std::uint64_t           s2          = 0;
        std::uint64_t           s3          = 0;
        for (unsigned long i : LONG_JUMP)
            for (int b = 0; b < 64; b++) {
                if (i & UINT64_C(1) << b) {
                    s0 ^= m_state[0];
                    s1 ^= m_state[1];
                    s2 ^= m_state[2];
                    s3 ^= m_state[3];
                }
                next();
            }

        m_state[0] = s0;
        m_state[1] = s1;
        m_state[2] = s2;
        m_state[3] = s3;
    }
};

#endif  // CPP_LEARNING_XOSHIROPLUSPLUS_H
