# Warning: this project is still WIP the API might change in the future.

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

All the generators are compatible with the C++11 random number generation utilities, such as std::
uniform_int_distribution.

For performance reasons a `uniform` method that generates double is offered. This method is faster than using a
uniform_real_distribution.
Refer to: https://prng.di.unimi.it/#remarks for more information.

## NOTE:

The Vectorized versions use an internal cache of 256 elements, plus 4 elements*SIMD width. So the DRAM requirements are
higher than the original Xoshiro256++ implementation.
The state size of the scalar versions is the same as the original implementations

## Multi-threading and cluster environments

All the generators in the xoshiro family can be used in a multi-threaded environment.
While the generators are NOT thread safe by design, the constructors take two optional arguments:
`thread_id` and `cluster_id`. This allow to have independent streams of random numbers for each thread and node in the
cluster.

EXAMPLE:
```cpp
#include <xoshiro/xoshiro.hpp>

xorshiro::Xoshiro rng(42, 1, 2); //Seed 42, Thread 1, Cluster 2
// when using std::thread one can do:
xorshiro::Xoshiro rng(42, std::this_thread::get_id();
// OpenMP
xorshiro::Xoshiro rng(42, omp_get_thread_num();
// OpenMP and MPI
xorshiro::Xoshiro rng(42, omp_get_thread_num(), MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
```

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

| ns/op |             op/s | err% | ins/op | cyc/op |   IPC | bra/op | miss% | total | benchmark                             
|------:|-----------------:|-----:|-------:|-------:|------:|-------:|------:|------:|:--------------------------------------
|  5.68 |   176,181,011.90 | 0.3% |  23.00 |  17.53 | 1.312 |   0.00 | 81.1% |  0.07 | Reference Xorshiro UINT64             
|  1.13 |   884,115,030.37 | 1.1% |  12.52 |   3.50 | 3.580 |   1.01 |  0.4% |  0.01 | Vector Xorshiro UINT64                
|  5.75 |   173,988,265.88 | 0.1% |  24.00 |  17.76 | 1.352 |   0.00 | 81.0% |  0.08 | Scalar Xorshiro UINT64                
|  0.80 | 1,251,157,782.73 | 0.7% |   8.78 |   2.47 | 3.555 |   1.01 |  0.6% |  0.01 | Dispatch Xorshiro UINT64              
|  1.30 |   767,630,802.57 | 1.5% |  25.45 |   4.02 | 6.323 |   1.26 |  0.3% |  0.02 | MersenneTwister UINT64                
|  1.54 |   648,787,513.41 | 1.2% |  18.52 |   4.76 | 3.889 |   1.01 |  0.4% |  0.02 | Vector Xorshiro DOUBLE                
|  5.74 |   174,357,729.55 | 0.1% |  29.00 |  17.72 | 1.637 |   0.00 | 41.3% |  0.07 | Scalar Xorshiro DOUBLE                
|  1.13 |   882,822,705.75 | 1.2% |  14.78 |   3.49 | 4.235 |   1.01 |  0.4% |  0.01 | Dispatch Xorshiro DOUBLE              
|  6.55 |   152,688,649.59 | 0.2% |  31.52 |  20.23 | 1.558 |   3.51 | 14.4% |  0.08 | Vector Xorshiro std::random<double>   
|  8.16 |   122,491,714.48 | 0.1% |  41.00 |  25.22 | 1.626 |   2.50 | 20.0% |  0.10 | Scalar Xorshiro std::random<double>   
|  5.84 |   171,134,877.74 | 0.3% |  27.78 |  18.04 | 1.540 |   3.51 | 14.4% |  0.07 | Dispatch Xorshiro std::random<double> 
|  7.74 |   129,189,841.55 | 0.3% |  41.46 |  23.90 | 1.734 |   3.76 | 13.5% |  0.10 | MersenneTwister std::random<double>   

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