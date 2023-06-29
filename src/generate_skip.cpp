//
// Created by mbarbone on 6/23/23.
//

#include <NTL/GF2X.h>
#include <NTL/vec_GF2.h>

#include <iostream>

#include "VectorXoshiro/xoshiroPlusPlus.h"

constexpr auto jump_step = 64;
constexpr auto MEXP = 256; /* the dimension of the state space */
NTL::GF2X      phi;        /* phi is the minimal polynomial */
NTL::GF2X      g;          /* g(t) is used to store t^J mod phi(t) */

/* computes the minimal polynomial of the linear recurrence */
void comp_mini_poly(XoshiroPlusPlus& rng) {
    int          i;
    NTL::vec_GF2 v(NTL::INIT_SIZE, 2 * MEXP);
    for (i = 0; i < 2 * MEXP; i++) { v[i] = static_cast<long>(rng()>>63); }
    MinPolySeq(phi, v, MEXP);
}

/* computes the t^J mod phi(t) */
void comp_jump_rem() { PowerXMod(g, jump_step, phi); }

// generate main
int main() {
    const auto        seed = 123456;
    XoshiroPlusPlus   rng(seed);
    comp_mini_poly(rng);
    comp_jump_rem();
    for (auto i=MEXP-1; i>-1; i--){
        std::cout << NTL::coeff(g, i) << " ";
    }
    std::cout << std::endl;
}