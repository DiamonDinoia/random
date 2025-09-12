"""Benchmark XoshiroSIMD against common NumPy generators.

This script generates a large array of random numbers using several
`numpy.random` generators and reports the fastest time observed.  It provides a
rough comparison of throughput for XoshiroSIMD versus NumPy's default
implementations.
"""

from __future__ import annotations

import time
from typing import Callable, Dict

import numpy as np
import pyrandom


def _benchmark(name: str, factory: Callable[[], np.random.Generator], *, size: int = 100_000_000, repeat: int = 10) -> None:
    """Benchmark a single generator.

    Parameters
    ----------
    name: str
        Display name of the generator.
    factory: Callable[[], numpy.random.Generator]
        Factory returning a fresh generator instance.
    size: int
        Number of random floats to draw.
    repeat: int
        Number of times the benchmark is repeated. The best time is reported.
    """
    timings = []
    # preallocate output to avoid timing allocations
    out = np.empty(size, dtype=np.float64)
    for _ in range(repeat):
        rng = factory()
        # optional warmup to trigger JIT/dispatch caches, etc.
        rng.random(1, out=out[:1])
        start = time.perf_counter()
        rng.random(size, out=out)  # bulk fill via NumPy's path
        timings.append(time.perf_counter() - start)
    best = min(timings)
    rate = size / best
    print(f"{name:15s}: {best:8.3f} s  ({rate/1e6:6.2f} M samples/s)")


def main() -> None:
    generators: Dict[str, Callable[[], np.random.Generator]] = {
        "Philox": lambda: np.random.Generator(np.random.Philox(1234)),
        "XoshiroSIMD": lambda: pyrandom.XoshiroSIMD(1234),
        "PCG64": lambda: np.random.Generator(np.random.PCG64(1234)),
        "XoshiroNative": lambda: pyrandom.XoshiroNative(1234),
        "MT19937": lambda: np.random.Generator(np.random.MT19937(1234)),
        "default_rng": lambda: np.random.default_rng(1234),
    }
    for name, factory in generators.items():
        _benchmark(name, factory)


if __name__ == "__main__":
    main()
