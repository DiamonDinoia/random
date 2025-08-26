# pip install galois
import numpy as np
import galois

MASK64 = (1 << 64) - 1
GF = galois.GF(2)

def rotl64(x, k):
    return ((x << k) | (x >> (64 - k))) & MASK64

def next_state_u64(s):
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
    arr = np.array(s, dtype='<u8')
    by = arr.view(np.uint8)
    bits = np.unpackbits(by, bitorder='little')
    return bits.astype(np.uint8)

def vec_bits_to_words(bits):
    by = np.packbits(bits.astype(np.uint8), bitorder='little')
    words = np.frombuffer(by.tobytes(), dtype='<u8')
    return [int(w) for w in words]

def build_T_GF():
    T = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        w = i // 64
        b = i % 64
        s = [0, 0, 0, 0]
        s[w] = 1 << b
        sn = next_state_u64(s)
        T[:, i] = state_to_vec_bits(sn)
    return GF(T)  # promote to GF(2) array

T = build_T_GF()

# e0 basis vector
e0 = GF.Zeros(256)
e0[0] = 1

# Build Krylov matrix K = [e0, T e0, T^2 e0, ..., T^255 e0] over GF(2)
K = GF.Zeros((256, 256))
v = e0.copy()
for c in range(256):
    K[:, c] = v
    v = T @ v

# Invert over GF(2)
K_inv = np.linalg.inv(K)  # works on galois arrays

def gf2_matpow_vec(T, N, v):
    # T^N * v via repeated squaring over GF(2)
    out = v.copy()
    first = True
    M = T.copy()
    n = N
    while n:
        if n & 1:
            out = M @ out
            first = False
        n >>= 1
        if n:
            M = M @ M
    return v if first else out

def jump_words(N: int):
    vN = gf2_matpow_vec(T, N, e0)
    coeffs = K_inv @ vN                 # GF(2) vector of length 256
    coeffs_bits = np.array(coeffs, dtype=np.uint8)  # 0/1
    return vec_bits_to_words(coeffs_bits)

# Reference checks
REF_JUMP_2P128 = [0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c]
REF_JUMP_2P192 = [0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635]

assert jump_words(1 << 128) == REF_JUMP_2P128
assert jump_words(1 << 192) == REF_JUMP_2P192

gen_2p160 = jump_words(1 << 160)
print("2^160 =", [f"0x{w:016x}" for w in gen_2p160])
