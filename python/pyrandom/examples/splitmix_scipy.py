import numpy as np
from scipy import stats
import pyrandom


def main() -> None:
    rng = pyrandom.SplitMix(123)
    normals = stats.norm.rvs(size=5, random_state=rng)
    print("scipy normals:", normals)
    integers = stats.randint.rvs(0, 10, size=5, random_state=rng)
    print("scipy integers:", integers)

if __name__ == "__main__":
    main()
