"""
Compute jump-polynomial coefficients for xoshiro256's linear core using GF(2) linear algebra.

Overview:
- The xoshiro256 state transition (without the scrambler/output function) is linear over GF(2).
- Represent the 256-bit state as a vector over GF(2): v = [s0..s3] (each 64-bit, LSB-first).
- Build the 256x256 transition matrix T such that v_{n+1} = T * v_n (over GF(2)).
- Build the Krylov matrix K = [e0, T e0, T^2 e0, ..., T^255 e0], where e0 = bit 0 basis vector.
- Invert K to express any vector as a GF(2) combination of these columns.
- For a jump of N steps, the coefficient vector of the jump polynomial is:
      coeffs = K^{-1} * (T^N * e0)
  Packed into 4x uint64 words (LSB-first) to match the reference format.
- Validate against reference 2^128 and 2^192 jumps and also emit 2^160.
"""

import numpy as np

# 64-bit mask for word operations
MASK64 = (1 << 64) - 1


def rotl64(x, k):
    """Rotate-left for 64-bit words."""
    return ((x << k) | (x >> (64 - k))) & MASK64


def next_state_u64(s):
    """
    One step of the xoshiro256 state transition (linear core only, no scrambler).

    State s = [s0, s1, s2, s3] of four uint64 words.
    This is exactly the same linear update used by xoshiro256**/++ minus the output mixing.
    """
    s0, s1, s2, s3 = s
    t = (s1 << 17) & MASK64
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = rotl64(s3, 45)
    return [s0, s1, s2, s3]


def state_to_vec_bits(s):
    """
    Convert [s0, s1, s2, s3] -> 256-bit GF(2) vector (uint8 of shape (256,)),
    using little-endian bit order within each 64-bit word (LSB-first).
    """
    arr = np.array(s, dtype='<u8')  # little-endian uint64 words
    by = arr.view(np.uint8)  # raw bytes
    bits = np.unpackbits(by, bitorder='little')
    return bits.astype(np.uint8)  # entries are 0 or 1


def vec_bits_to_words(bits):
    """
    Pack a 256-bit coefficient vector (LSB-first) into 4x uint64 little-endian words,
    matching the format used by xoshiro jump constants.
    """
    by = np.packbits(bits.astype(np.uint8), bitorder='little')
    words = np.frombuffer(by.tobytes(), dtype='<u8')  # 4 words
    return [int(w) for w in words]


def build_T():
    """
    Build the 256x256 GF(2) transition matrix T such that:
      state_vector(next_state(s)) == T * state_vector(s)
    Columns of T are images of the standard basis vectors under next_state_u64.
    """
    T = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        w = i // 64  # which 64-bit word (0..3)
        b = i % 64  # which bit inside that word (0..63)
        s = [0, 0, 0, 0]
        s[w] = 1 << b  # single 1-bit in the i-th basis position
        sn = next_state_u64(s)  # apply linear transition
        T[:, i] = state_to_vec_bits(sn)  # resulting vector is column i of T
    return T


def gf2_inv(A):
    """
    Invert a square binary matrix over GF(2) via Gauss-Jordan elimination.
    Row operations are XORs; pivot search picks any row with a 1 in the pivot column.
    """
    n = A.shape[0]
    A = A.copy()
    I = np.eye(n, dtype=np.uint8)
    for col in range(n):
        # Find a pivot at or below the current row.
        piv = np.where(A[col:, col] == 1)[0]
        if piv.size == 0:
            raise RuntimeError("Matrix not invertible over GF(2)")
        r = piv[0] + col
        # Swap pivot row into place.
        if r != col:
            A[[col, r]] = A[[r, col]]
            I[[col, r]] = I[[r, col]]
        # Eliminate the column in all other rows.
        rows = np.where((A[:, col] == 1) & (np.arange(n) != col))[0]
        if rows.size:
            A[rows] ^= A[col]
            I[rows] ^= I[col]
    return I


def gf2_matmul(A, B):
    """
    Matrix multiply over GF(2). Uses integer matmul followed by &1 to reduce mod 2.
    This is correct because we only need the parity of dot products.
    """
    return (A @ B) & 1


def gf2_matpow_vec(T, N, v):
    """
    Compute T^N * v over GF(2) using repeated squaring, but only keep a vector.
    Returns v unchanged if N == 0.
    """
    out = v.copy()
    first = True
    M = T.copy()
    n = N
    while n:
        if n & 1:
            out = gf2_matmul(M, out)
            first = False
        n >>= 1
        if n:
            M = gf2_matmul(M, M)
    return v if first else out


# Build T (transition), K (Krylov basis), and K^{-1}.
T = build_T()

# e0 is the 256-bit vector with only bit 0 set (s0's LSB).
e0_state = [1, 0, 0, 0]
v = state_to_vec_bits(e0_state)

# K's columns are e0, T e0, T^2 e0, ..., T^255 e0.
# This basis lets us express any T^N e0 as a linear combination of these columns.
K = np.zeros((256, 256), dtype=np.uint8)
for c in range(256):
    K[:, c] = v
    v = gf2_matmul(T, v)

K_inv = gf2_inv(K)

# Single-basis vector to select the first column in K (bit 0).
e0 = np.zeros(256, dtype=np.uint8)
e0[0] = 1


def jump_words(N: int):
    """
    Compute the 256-bit jump polynomial coefficients (packed into 4x uint64)
    for advancing the state by exactly N steps.

    Steps:
      - vN = T^N * e0 (desired column if we extended K infinitely).
      - coeffs = K^{-1} * vN gives GF(2) coefficients for the jump polynomial.
      - Pack coeffs into 4 little-endian uint64 words (LSB-first per xoshiro's format).
    """
    vN = gf2_matpow_vec(T, N, e0)
    coeffs = gf2_matmul(K_inv, vN)  # 256 coefficients (LSB-first order)
    return vec_bits_to_words(coeffs)  # 4x uint64 words


# Reference constants from xoshiro256++ (Blackman/Vigna) for verification.
REF_JUMP_2P128 = [0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c]
REF_JUMP_2P192 = [0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635]

# Generate and check against references.
gen_2p128 = jump_words(1 << 128)
gen_2p192 = jump_words(1 << 192)

assert gen_2p128 == REF_JUMP_2P128, "Mismatch for 2^128 jump"
assert gen_2p192 == REF_JUMP_2P192, "Mismatch for 2^192 long jump"
print("2^128 and 2^192 jumps match the reference constants.")


def fmt(name, words):
    """Pretty-printer for the 4x uint64 jump constants."""
    print(f"{name} = {{ " + ", ".join(f"0x{w:016x}" for w in words) + " }")


# Also show 2^160 jump constants derived by this method.
gen_2p160 = jump_words(1 << 160)

fmt("Generated 2^128", gen_2p128)
fmt("Generated 2^160", gen_2p160)
fmt("Generated 2^192", gen_2p192)
