from pyrandom import Xoshiro
from scipy import stats


def main():
    rng = Xoshiro(2024)
    samples = stats.norm.rvs(size=5, random_state=rng)
    print(samples)


if __name__ == "__main__":
    main()
