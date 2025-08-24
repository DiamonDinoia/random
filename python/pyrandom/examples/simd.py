
from pyrandom import XoshiroSIMD


def main():
    rng = XoshiroSIMD(42)
    print(rng.integers(0, 10, size=5))


if __name__ == "__main__":
    main()
