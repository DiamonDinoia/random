#pragma once

#include "ecrypt-portable.h"

// Wrapper that calls the C reference implementation of ChaCha20
// (https://cr.yp.to/chacha.html). The main implementation function,
// `salsa20_wordtobyte`, of the reference implementation has internal linkage
// only, so we rexpose it via this wrapper.
// 
// Reference implementation obtained from:
// https://cr.yp.to/streamciphers/timings/estreambench/submissions/salsa20/chacha20/ref/chacha.c
void chacha20_ref_impl_wrapper(u8 output[64],const u32 input[16]);
