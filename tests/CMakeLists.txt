CPMAddPackage(
        NAME catch2
        GITHUB_REPOSITORY catchorg/Catch2
        VERSION 2.13.7
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM
)

CPMAddPackage(
        NAME nanobench
        GITHUB_REPOSITORY martinus/nanobench
        VERSION 4.3.11
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM
)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
set(TEST_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)

# Download reference implementations here. The ChaCha reference implementation
# requires multiple files because it transitively includes macros from them
# (see https://cr.yp.to/chacha.html). Although the linked page links to ChaCha8
# implementations, we download ChaCha20 versions instead. The files
# ecrypt-{portable,config,machine}.h are taken from the Salsa20 reference
# implementation rather than ChaCha20, since no ChaCha20 versions of these
# appear to exist. Since ChaCha20 is a subdir in the Salsa20 directory in
# EBACS, and given ChaCha20 is structured very similar to Salsa20, I can only
# assume these header files are also the ones used for ChaCha20.
file(MAKE_DIRECTORY ${TEST_INCLUDE_DIR})
file(DOWNLOAD https://prng.di.unimi.it/splitmix64.c ${TEST_INCLUDE_DIR}/splitmix64.c)
file(DOWNLOAD https://prng.di.unimi.it/xoshiro256plusplus.c ${TEST_INCLUDE_DIR}/xoshiro256plusplus.c)
file(DOWNLOAD https://cr.yp.to/streamciphers/timings/estreambench/submissions/salsa20/chacha20/ref/chacha.c ${TEST_INCLUDE_DIR}/chacha.c)
file(DOWNLOAD https://cr.yp.to/streamciphers/timings/estreambench/submissions/salsa20/chacha20/ref/ecrypt-sync.h ${TEST_INCLUDE_DIR}/ecrypt-sync.h)
file(DOWNLOAD https://cr.yp.to/snuffle/ecrypt-portable.h ${TEST_INCLUDE_DIR}/ecrypt-portable.h)
file(DOWNLOAD https://cr.yp.to/snuffle/ecrypt-config.h ${TEST_INCLUDE_DIR}/ecrypt-config.h)
file(DOWNLOAD https://cr.yp.to/snuffle/ecrypt-machine.h ${TEST_INCLUDE_DIR}/ecrypt-machine.h)

include(Catch)
include(CTest)
enable_testing()

add_executable(testSplitMix splitmix_tests.cpp)
target_link_libraries(testSplitMix PRIVATE random Catch2::Catch2)
target_include_directories(testSplitMix PRIVATE ${TEST_INCLUDE_DIR})
catch_discover_tests(testSplitMix)

add_executable(testXoshiro test_xoshiro.cpp)
target_link_libraries(testXoshiro PRIVATE random Catch2::Catch2)
target_include_directories(testXoshiro PRIVATE ${TEST_INCLUDE_DIR})
catch_discover_tests(testXoshiro)

add_executable(testXoshiroSIMD test_xoshiro_simd.cpp)
target_link_libraries(testXoshiroSIMD PRIVATE random Catch2::Catch2)
target_include_directories(testXoshiroSIMD PRIVATE ${TEST_INCLUDE_DIR} xsimd::xsimd)
catch_discover_tests(testXoshiroSIMD)

add_executable(testChaCha test_chacha.cpp)
target_link_libraries(testChaCha PRIVATE random Catch2::Catch2)
target_include_directories(testChaCha PRIVATE ${TEST_INCLUDE_DIR})
# ChaCha20's reference implementation's constants are set like
#       static const char sigma[16] = "expand 2-byte k"
# Causes error without -fpermissive because it doesn't fit null terminator.
target_compile_options(testChaCha PRIVATE -fpermissive -w)
catch_discover_tests(testChaCha)

add_executable(benchmarks benchmarks.cpp)
target_link_libraries(benchmarks PRIVATE random nanobench)
target_include_directories(benchmarks PRIVATE ${TEST_INCLUDE_DIR} xsimd::xsimd)
target_compile_options(benchmarks PRIVATE ${COMPILE_OPTIONS})
target_link_options(benchmarks PRIVATE ${LINK_OPTIONS})

if (MARCH_NATIVE)
    target_compile_options(benchmarks PRIVATE -march=native)
endif ()

# Add a custom target to run the benchmarks
add_custom_target(benchmark
    COMMAND benchmarks
    DEPENDS benchmarks
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running benchmarks..."
    VERBATIM
)


if (MARCH_NATIVE)
    target_compile_options(benchmarks PRIVATE -march=native)
endif ()
