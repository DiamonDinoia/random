import threading
import numpy as np
from typing import Optional

from pyrandom_ext import (
    # C++ factories (capsule -> NumPy BitGenerator)
    create_splitmix_bit_generator,
    create_xoshiro_bit_generator,                  # Xoshiro (scalar)
    create_bit_generator as create_xoshiro_simd_bit_generator,  # XoshiroSIMD
    create_xoshiro_native_bit_generator,           # XoshiroNative (overloaded)
)

class _CapsuleBitGen:
    """Minimal adapter: exactly what numpy.random.Generator expects."""
    __slots__ = ("capsule", "lock")

    def __init__(self, capsule):
        self.capsule = capsule
        self.lock = threading.Lock()


def SplitMix(seed: int) -> np.random.Generator:
    """np.random.Generator seeded with SplitMix."""
    cap = create_splitmix_bit_generator(seed)
    return np.random.Generator(_CapsuleBitGen(cap))


def Xoshiro(seed: int) -> np.random.Generator:
    """np.random.Generator seeded with scalar Xoshiro."""
    cap = create_xoshiro_bit_generator(seed)
    return np.random.Generator(_CapsuleBitGen(cap))


def XoshiroSIMD(seed: int) -> np.random.Generator:
    """np.random.Generator seeded with XoshiroSIMD."""
    cap = create_xoshiro_simd_bit_generator(seed)
    return np.random.Generator(_CapsuleBitGen(cap))


def XoshiroNative(
        seed: int,
        thread: Optional[int] = None,
        cluster: Optional[int] = None,
) -> np.random.Generator:
    """np.random.Generator seeded with XoshiroNative (supports thread/cluster)."""
    if thread is None:
        cap = create_xoshiro_native_bit_generator(seed)
    elif cluster is None:
        cap = create_xoshiro_native_bit_generator(seed, thread)
    else:
        cap = create_xoshiro_native_bit_generator(seed, thread, cluster)
    return np.random.Generator(_CapsuleBitGen(cap))


__all__ = [
    "SplitMix",
    "Xoshiro",
    "XoshiroSIMD",
    "XoshiroNative",
]
