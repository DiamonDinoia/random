import numpy as np
import threading

from pyrandom_ext import (
    SplitMix64,
    Xoshiro as XoshiroRNG,
    VectorXoshiro as VectorXoshiroRNG,
    VectorXoshiroNative as VectorXoshiroNativeRNG,
    create_xoshiro_bit_generator,
    create_vector_bit_generator,
)


class XoshiroBitGenerator:
    """Wrapper carrying the Xoshiro bit generator capsule."""
    def __init__(self, capsule):
        self.capsule = capsule
        self.lock = threading.Lock()


class VectorXoshiroBitGenerator:
    """Wrapper carrying the VectorXoshiro bit generator capsule."""
    def __init__(self, capsule):
        self.capsule = capsule
        self.lock = threading.Lock()


def Xoshiro(seed: int) -> np.random.Generator:
    """Return a numpy.random.Generator seeded with Xoshiro."""
    capsule = create_xoshiro_bit_generator(seed)
    bitgen = XoshiroBitGenerator(capsule)
    return np.random.Generator(bitgen)


def VectorXoshiro(seed: int) -> np.random.Generator:
    """Return a numpy.random.Generator seeded with VectorXoshiro."""
    capsule = create_vector_bit_generator(seed)
    bitgen = VectorXoshiroBitGenerator(capsule)
    return np.random.Generator(bitgen)


__all__ = [
    "SplitMix64",
    "XoshiroRNG",
    "VectorXoshiroRNG",
    "VectorXoshiroNativeRNG",
    "Xoshiro",
    "VectorXoshiro",
]
