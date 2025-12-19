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

file(MAKE_DIRECTORY ${TEST_INCLUDE_DIR})
file(DOWNLOAD https://prng.di.unimi.it/splitmix64.c ${TEST_INCLUDE_DIR}/splitmix64.c)
file(DOWNLOAD https://prng.di.unimi.it/xoshiro256plusplus.c ${TEST_INCLUDE_DIR}/xoshiro256plusplus.c)

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
