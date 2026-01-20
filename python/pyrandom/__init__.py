import threading
import numpy as np
from typing import Optional

from pyrandom_ext import (
    # C++ factories (capsule -> NumPy BitGenerator)
    create_splitmix_bit_generator,
    create_xoshiro_bit_generator,                  # Xoshiro (scalar)
    create_bit_generator as create_xoshiro_simd_bit_generator,  # XoshiroSIMD
    create_xoshiro_native_bit_generator,           # XoshiroNative (overloaded)
    # Core persistent RNG with bulk-fill
    XoshiroSIMD as _CoreXoshiroSIMD,
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
    """High-throughput RNG with XoshiroSIMD.

    - Fast path: bulk `float64` fills via the persistent core (`fill_uniform`).
    - Fallback: delegate to a persistent NumPy `Generator` backed by the same
      XoshiroSIMD bitgenerator for non-float64 or non-contiguous/out-of-shape cases.
    """
    class _XoshiroSIMDGen:
        __slots__ = ("_core", "_np")

        def __init__(self, seed: int):
            self._core = _CoreXoshiroSIMD(seed)
            # Persistent fallback Generator (avoid re-seeding/re-allocating per call)
            cap = create_xoshiro_simd_bit_generator(seed)
            self._np = np.random.Generator(_CapsuleBitGen(cap))

        def random(self, size, dtype=np.float64, out: Optional[np.ndarray] = None):
            # Normalize size to an int or tuple
            shape = size if isinstance(size, tuple) else (int(size),)
            # Fast path for integer dtypes: reinterpret storage as uint64 and fill
            if np.issubdtype(np.dtype(dtype), np.integer):
                if out is None:
                    out = np.empty(shape, dtype=dtype, order='K')
                if (
                    isinstance(out, np.ndarray)
                    and out.size == int(np.prod(shape))
                    and (out.flags.c_contiguous or out.flags.f_contiguous)
                ):
                    flat = out.ravel(order='K')
                    bytes_view = flat.view(np.uint8)
                    total = bytes_view.size
                    q = (total // 8) * 8
                    if q:
                        u64_view = bytes_view[:q].view(np.uint64)
                        self._core._fill_uint64(u64_view)
                    r = total - q
                    if r:
                        tmp = np.empty(1, dtype=np.uint64)
                        self._core._fill_uint64(tmp)
                        bytes_view[q:q+r] = tmp.view(np.uint8)[:r]
                    return out
                # Fallback to NumPy for non-contiguous storage
                return self._np.integers(0, 2**64, size=shape, dtype=dtype)
            if dtype is np.float64:
                if out is None:
                    out = np.empty(shape, dtype=np.float64, order='K')
                # Fast path: any contiguous float64 buffer (C or F), size must match
                if (
                    isinstance(out, np.ndarray)
                    and out.dtype == np.float64
                    and out.size == int(np.prod(shape))
                    and (out.flags.c_contiguous or out.flags.f_contiguous)
                ):
                    # Use a 1D view over the existing storage
                    flat = out.ravel(order='K')
                    # Use internal fast path and avoid NumPy's contiguity requirement
                    self._core._fill_uniform(flat)
                    return out
            # Fallback: delegate to persistent NumPy Generator for other dtypes/shapes
            return self._np.random(size, dtype=dtype, out=out)

        def __getattr__(self, name):
            # Delegate any other methods/attrs to the underlying NumPy Generator
            return getattr(self._np, name)

    return _XoshiroSIMDGen(seed)


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
