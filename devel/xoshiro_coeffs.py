# pip install galois
import galois
import numpy as np

# -----------------------------------------------------------------------------
# Model the xoshiro256 linear core over GF(2) and compute jump polynomials.
#
# Idea:
# - The 256-bit state is a vector over GF(2) with bit i corresponding to x^i.
# - The linear state transition defines a 256x256 matrix T over GF(2).
# - Let e0 be the vector with bit 0 set (LSB of s0). The Krylov matrix
#     K = [e0, T e0, T^2 e0, ..., T^255 e0]
#   is invertible because e0 is a cyclic vector for this generator.
# - For a jump of N steps, the coefficient vector is:
#     coeffs = K^{-1} * (T^N * e0)
#   These 256 coefficients (0/1) are then packed into the 4×uint64 jump words
#   in little-endian (LSB-first) format to match xoshiro's reference constants.
# -----------------------------------------------------------------------------

MASK64 = (1 << 64) - 1
GF = galois.GF(2)  # Binary field for all our linear algebra

def rotl64(x, k):
    """Rotate-left for 64-bit words (mask to keep within 64 bits)."""
    return ((x << k) | (x >> (64 - k))) & MASK64

def next_state_u64(s):
    """
    One step of the xoshiro256 state transition (linear core only, no scrambler).
    This matches the linear mixing used by xoshiro256**/++ minus the output.
    State s = [s0, s1, s2, s3], each a uint64.
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
    Convert [s0, s1, s2, s3] into a 256-bit GF(2) vector (uint8[256]) using
    little-endian bit order within each 64-bit word (LSB-first).
    This ordering matches xoshiro's published jump constants format.
    """
    arr = np.array(s, dtype='<u8')  # 4×uint64 (little-endian)
    by = arr.view(np.uint8)  # 32 raw bytes
    bits = np.unpackbits(by, bitorder='little')  # 256 bits, LSB-first per 64-bit word
    return bits.astype(np.uint8)  # values are 0 or 1

def vec_bits_to_words(bits):
    """
    Pack a 256-bit coefficient vector (index i = coeff of x^i) into 4×uint64
    words in little-endian (LSB-first) order, identical to xoshiro jump constants.
    """
    by = np.packbits(bits.astype(np.uint8), bitorder='little')
    words = np.frombuffer(by.tobytes(), dtype='<u8')  # 4 little-endian uint64s
    return [int(w) for w in words]

def build_T_GF():
    """
    Build the 256×256 GF(2) transition matrix T such that:
        vec(next_state_u64(s)) == T @ vec(s)
    We fill T column-by-column by applying next_state_u64 to each standard basis
    vector (a single 1-bit at position i).
    """
    T = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        w = i // 64  # select which 64-bit word (0..3)
        b = i % 64  # select which bit within that word (0..63)
        s = [0, 0, 0, 0]
        s[w] = 1 << b  # basis vector: only bit i set
        sn = next_state_u64(s)  # apply one linear step
        T[:, i] = state_to_vec_bits(sn)  # resulting image is column i
    return GF(T)  # promote to a GF(2) matrix


# Build the transition matrix once
T = build_T_GF()

# e0 is the "cyclic" basis vector: bit 0 set (LSB of s0)
e0 = GF.Zeros(256);
e0[0] = 1

# Build the 256×256 Krylov matrix K = [e0, T e0, T^2 e0, ..., T^255 e0].
# This spans the full state space for xoshiro256's linear core, so K is invertible.
K = GF.Zeros((256, 256))
v = e0.copy()
for c in range(256):
    K[:, c] = v
    v = T @ v

# Invert K over GF(2). The galois arrays integrate with NumPy's linear algebra
# so np.linalg.inv works and returns a GF(2) matrix.
K_inv = np.linalg.inv(K)

def gf2_matpow_vec(T, N, v):
    """
    Compute T^N @ v over GF(2) using exponentiation by squaring.
    This is O(log N) matrix multiplications (still with 256x256 matrices).
    Returns v when N == 0.
    """
    out = v.copy()
    first = True  # remember if we never multiplied (N == 0)
    M = T.copy()
    n = N
    while n:
        if n & 1:
            out = M @ out
            first = False
        n >>= 1
        if n:
            M = M @ M  # square the base matrix
    return v if first else out

def jump_words(N: int):
    """
    Return the 4×uint64 jump polynomial coefficients for a jump of exactly N steps.
    Steps:
      1) Compute vN = T^N @ e0  (column we'd get if the Krylov basis extended infinitely).
      2) Solve coeffs = K^{-1} @ vN  to express vN as a GF(2) combination of columns of K.
      3) Pack those 256 coeff bits into 4 LSB-first uint64s matching xoshiro's format.
    """
    vN = gf2_matpow_vec(T, N, e0)
    coeffs = K_inv @ vN  # 256-length GF(2) vector
    coeffs_bits = np.array(coeffs, dtype=np.uint8)  # convert to plain 0/1
    return vec_bits_to_words(coeffs_bits)


# Reference jump constants from Blackman/Vigna for verification.
REF_JUMP_2P128 = [0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c]
REF_JUMP_2P192 = [0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635]

# Sanity-check: our computed jumps must match the published constants.
assert jump_words(1 << 128) == REF_JUMP_2P128
assert jump_words(1 << 192) == REF_JUMP_2P192

# Also emit 2^160 derived by the same method.
gen_2p160 = jump_words(1 << 160)
print("2^160 =", ' '.join([f"0x{w:016x}" for w in gen_2p160]))

# Notes:
# - Building T and inverting K are one-time costs (~256^3 ops over GF(2)).
# - After K_inv is known, each jump is O(log N) mat-mults via gf2_matpow_vec().
# - If you need many different N, consider switching to the polynomial method:
#     compute the characteristic/minimal polynomial P(x) once, then use
#     x^N mod P(x) to get the coefficients without touching 256×256 matrices.
