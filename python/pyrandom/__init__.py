import numpy as np
import threading

from pyrandom_ext import (
    SplitMix,
    XoshiroSIMD as XoshiroSIMDRNG,
    XoshiroNative as XoshiroNativeRNG,
    create_bit_generator,
    create_splitmix_bit_generator,
)


class XoshiroBitGenerator:
    """Wrapper carrying the Xoshiro bit generator capsule."""
    def __init__(self, capsule):
        self.capsule = capsule
        self.lock = threading.Lock()


def SplitMix(seed: int) -> np.random.Generator:
    """Return a numpy.random.Generator seeded with SplitMix."""
    capsule = create_splitmix_bit_generator(seed)
    bitgen = XoshiroBitGenerator(capsule)
    return np.random.Generator(bitgen)


def Xoshiro(seed: int) -> np.random.Generator:
    """Return a numpy.random.Generator seeded with XoshiroSIMD."""
    capsule = create_bit_generator(seed)
    bitgen = XoshiroBitGenerator(capsule)
    return np.random.Generator(bitgen)


def XoshiroSIMD(seed: int) -> np.random.Generator:
    """Return a numpy.random.Generator seeded with XoshiroSIMD."""
    return Xoshiro(seed)


__all__ = [
    "SplitMix",
    "XoshiroSIMDRNG",
    "XoshiroNativeRNG",
    "Xoshiro",
    "XoshiroSIMD",
]
