cmake_minimum_required(VERSION 3.16)

project(cpp_learning)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(DCMAKE_RULE_MESSAGES OFF)

# version of the libraries to be used
set(CPM_DOWNLOAD_VERSION 0.38.1)
set(EIGEN_DOWNLOAD_VERSION 3.4.0)
set(HIGHWAY_DOWNLOAD_VERSION 1.0.4)

option(GENERATE_SKIP "Generate skip table" OFF)
option(ENABLE_TESTS "Enable tests" ON)

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
    add_compile_options(-Wall -Wextra -pedantic -fno-omit-frame-pointer -g -O0 -Wno-gnu-zero-variadic-macro-arguments)

endif ()
if (CMAKE_BUILD_TYPE MATCHES Release)
    add_compile_options(-O3)
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
        VERSION 1.8.3
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM YES
        OPTIONS
        "BENCHMARK_ENABLE_TESTNG Off"
        "BENCHMARK_USE_LIBCXX Off"
        "HAVE_STD_REGEX On"
        "RUN_HAVE_STD_REGEX ON"
)


add_library(vectorXoshiro INTERFACE)
target_include_directories(vectorXoshiro INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(vectorXoshiro INTERFACE cxx_std_17) # This means at least C++17
target_link_libraries(vectorXoshiro INTERFACE hwy)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

if (GENERATE_SKIP)
    find_library(NTL_LIB ntl)
    if (NOT NTL_LIB)
        message(FATAL_ERROR "ntl library not found.")
    endif ()

    find_library(GMP_LIB gmp)
    if (NOT GMP_LIB)
        message(FATAL_ERROR "gmp library not found.")
    endif ()

    add_executable(skip src/generate_skip.cpp)
    target_link_libraries(skip PUBLIC ${NTL_LIB} ${GMP_LIB} vectorXoshiro)
endif ()