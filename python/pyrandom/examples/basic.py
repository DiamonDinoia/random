
from pyrandom import Xoshiro


def main():
    rng = Xoshiro(1234)
    print(rng.integers(0, 10, size=5))


if __name__ == "__main__":
    main()
