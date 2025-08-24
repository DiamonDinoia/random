import numpy as np
import pyrandom


def main() -> None:
    rng = pyrandom.XoshiroSIMD(123)
    a = rng.random((3, 4))
    print("uniform:\n", a)

    b = np.empty((1000,), dtype=np.float32)
    rng.random(b.shape, dtype=np.float32, out=b)
    print("uniform float32 first 5:", b[:5])

    c = rng.integers(0, 10, size=(2, 5))
    print("integers:\n", c)

    d = rng.normal(0.0, 1.0, size=(3, 3))
    print("normal:\n", d)

if __name__ == "__main__":
    main()
