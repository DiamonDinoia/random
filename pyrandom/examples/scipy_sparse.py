from pyrandom import VectorXoshiro
from scipy import sparse


def main():
    rng = VectorXoshiro(2024)
    mat = sparse.random(3, 3, density=0.5, random_state=rng)
    print(mat.toarray())


if __name__ == "__main__":
    main()
