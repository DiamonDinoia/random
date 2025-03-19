# Public API Documentation

## Overview

This project provides a set of utilities for random number generation using the Xoshiro family of algorithms.
It implements the original Xoshiro256++ algorithm, SplitMix64 from Sebastiano Vigna, and the VectorXoshiro algorithms
from David Blackman and Sebastiano Vigna.

There are four public classes available for use:
SplitMix: A simple random number generator based on the SplitMix64 algorithm.
Xoshiro: A random number generator based on the Xoshiro256++ algorithm.
VectorXoshiro: A vectorized random number generator based on the Xoshiro256++ algorithm. That uses cpu_id dispatching to
select the best implementation for the current CPU.
VectorXoshiroNative: A vectorized random number generator that should be used when compiling with -march=native, -mcpu
for best performance.


All the generators are compatible with the C++11 random number generation utilities, such as std::uniform_int_distribution.

For performance reasons a `uniform` method that generates double is offered. This method is faster than using a uniform_real_distribution.
Refer to: https://prng.di.unimi.it/#remarks for more information.

## NOTE:

The Vectorized versions use an internal cache of 256 elements, plus 4 elements*SIMD width. So the DRAM requirements are
higher than the original Xoshiro256++ implementation.
The state size of the scalar versions is the same as the original implementations

## Build Instructions

To build the project, ensure you have CMake and a compatible C++ compiler installed. Follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/DiamonDinoia/VectorXoshiro.git
    cd VectorXoshiro
    ```

2. Create a build directory and navigate into it:
    ```sh
    mkdir build
    cd build
    ```

3. Run CMake to configure the project:
    ```sh
    cmake ..
    ```

4. Build the project:
    ```sh
    make
    ```

## Running Tests

The project includes several test executables. To run the tests, use the following commands from the build directory:

- `testSplitMix`:
    ```sh
    ctest -R testSplitMix
    ```

- `testXoshiro`:
    ```sh
    ctest -R testXoshiro
    ```

- `testVectorXoshiro`:
    ```sh
    ctest -R testVectorXoshiro
    ```

The tests will also check the output against the official Xoshiro256++ reference implementation. Which cmake downloads
compile and links against.

## Benchmarking

To run the benchmarks, execute the `benchmarks` binary:


```sh
./benchmarks

```

Example output is:

``` 
|               ns/op |                op/s |    err% |     total | benchmark
|--------------------:|--------------------:|--------:|----------:|:----------
|                1.64 |      608,409,415.30 |    0.3% |      0.02 | `Reference Xorshiro UINT64`
|                1.51 |      660,321,505.94 |    1.8% |      0.02 | `Vector Xorshiro UINT64`
|                0.93 |    1,071,886,459.17 |    0.3% |      0.01 | `Scalar Xorshiro UINT64`
|                0.92 |    1,081,366,586.86 |    0.2% |      0.01 | `Dispatch Xorshiro UINT64`
|                1.40 |      713,584,566.65 |    0.8% |      0.02 | `MersenneTwister UINT64`
|                1.86 |      536,939,824.28 |    0.3% |      0.02 | `Vector Xorshiro DOUBLE`
|                1.18 |      849,719,805.72 |    4.2% |      0.02 | `Scalar Xorshiro DOUBLE`
|                1.19 |      839,624,610.43 |    1.1% |      0.02 | `Dispatch Xorshiro DOUBLE`
|                6.53 |      153,221,543.16 |    0.9% |      0.08 | `Vector Xorshiro std::random<double>`
|                5.30 |      188,583,167.48 |    0.2% |      0.07 | `Scalar Xorshiro std::random<double>`
|                5.67 |      176,323,347.34 |    0.3% |      0.07 | `Dispatch Xorshiro std::random<double>`
|                7.58 |      131,887,676.12 |    1.6% |      0.10 | `MersenneTwister std::random<double>`
```


API Usage
Xoshiro256++ Example
Here is an example of how to use the Xoshiro256++ generator in your code:

```cpp
#include <xoshiro/xoshiro.hpp>

int main() {
    const auto seed = std::random_device()();
    xoshiro::Xoshiro rng(seed);

    // Generate a random number
    uint64_t random_number = rng();

    // Generate a random floating-point number between 0 and 1
    double random_float = rng.uniform();
    
    // Generate a random floating-point number between 0 and 1 using std::uniform_real_distribution
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double random_float2 = dist(rng);
}
```

VectorXoshiro Example
Here is an example of how to use the VectorXoshiro generator in your code:

```cpp
#include <xoshiro/vector_xoshiro.hpp>

int main() {
    const auto seed = std::random_device()();
    xoshiro::VectorXoshiro rng(seed);

    // Generate a random number
    uint64_t random_number = rng();

    // Generate a random floating-point number between 0 and 1
    double random_float = rng.uniform();
    
    // Generate a random floating-point number between 0 and 1 using std::uniform_real_distribution
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double random_float2 = dist(rng);
}
```

VectorXoshiroNative Example
Here is an example of how to use the VectorXoshiroNative generator in your code:

```cpp
#include <xoshiro/vector_xoshiro_native.hpp>

int main() {
    const auto seed = std::random_device()();
    xoshiro::VectorXoshiroNative rng(seed);

    // Generate a random number
    uint64_t random_number = rng();

    // Generate a random floating-point number between 0 and 1
    double random_float = rng.uniform();
    
    // Generate a random floating-point number between 0 and 1 using std::uniform_real_distribution
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double random_float2 = dist(rng);
}

```