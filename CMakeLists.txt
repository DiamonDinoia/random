cmake_minimum_required(VERSION 3.16)

project(cpp_learning)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(DCMAKE_RULE_MESSAGES OFF)

# version of the libraries to be used
set(CPM_DOWNLOAD_VERSION 0.38.1)
set(EIGEN_DOWNLOAD_VERSION 3.4.0)
set(HIGHWAY_DOWNLOAD_VERSION 1.0.4)


if (CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif (DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else ()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif ()
if (NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
            https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
            ${CPM_DOWNLOAD_LOCATION}
            )
endif ()
include(${CPM_DOWNLOAD_LOCATION})



if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_options(-fstack-protector-all)
    add_compile_options(-Wall -Wextra -pedantic)
endif ()
if (CMAKE_BUILD_TYPE MATCHES Release)
    add_compile_options(-ffp-contract=fast )
    add_compile_options(-freciprocal-math -fmerge-all-constants -fno-math-errno)
    add_compile_options(-march=native)
endif ()


CPMAddPackage(
        NAME Highway
        GIT_REPOSITORY https://github.com/google/highway.git
        GIT_TAG ${HIGHWAY_DOWNLOAD_VERSION}
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM YES
        OPTIONS
            "HWY_FORCE_STATIC_LIBS Yes"
)

CPMAddPackage(
        NAME Benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        VERSION 1.7.1
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM YES
        OPTIONS
            "BENCHMARK_ENABLE_TESTNG Off"
            "BENCHMARK_USE_LIBCXX On"
)

add_library(vectorXoroshiro INTERFACE)
target_include_directories(vectorXoroshiro INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(vectorXoroshiro INTERFACE cxx_std_17) # This means at least C++17

add_subdirectory(tests)