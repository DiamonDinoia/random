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
include(Catch)
include(CTest)
enable_testing()

add_executable(testSplitMix splitmix_tests.cpp)
target_link_libraries(testSplitMix PRIVATE random Catch2::Catch2)
catch_discover_tests(testSplitMix)

add_executable(testXoshiro test_xoshiro.cpp)
target_link_libraries(testXoshiro PRIVATE random Catch2::Catch2)
catch_discover_tests(testXoshiro)

add_executable(testXoshiroSIMD test_xoshiro_simd.cpp)
target_link_libraries(testXoshiroSIMD PRIVATE random Catch2::Catch2)
catch_discover_tests(testXoshiroSIMD)

add_executable(benchmarks benchmarks.cpp)
target_link_libraries(benchmarks PRIVATE random nanobench)

if (MARCH_NATIVE)
    target_compile_options(benchmarks PRIVATE -march=native)
endif ()
