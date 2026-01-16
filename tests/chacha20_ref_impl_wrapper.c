#include "chacha.c"

void chacha20_ref_impl_wrapper(u8 output[64],const u32 input[16]) {
  salsa20_wordtobyte(output, input);
}

