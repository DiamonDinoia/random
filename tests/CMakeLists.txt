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

include(CTest)
enable_testing()

set(TEST_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)

file(MAKE_DIRECTORY ${TEST_INCLUDE_DIR})
file(DOWNLOAD https://prng.di.unimi.it/splitmix64.c ${TEST_INCLUDE_DIR}/splitmix64.c)
file(DOWNLOAD https://prng.di.unimi.it/xoshiro256plusplus.c ${TEST_INCLUDE_DIR}/xoshiro256plusplus.c)

add_executable(testSplitMix split_mix_tests.cpp)
target_link_libraries(testSplitMix PRIVATE vectorXoshiro Catch2::Catch2)
target_include_directories(testSplitMix PRIVATE SYSTEM ${TEST_INCLUDE_DIR})
add_test(NAME testSplitMix COMMAND testSplitMix)

add_executable(testXoshiro test_xoshiro.cpp)
target_link_libraries(testXoshiro PRIVATE vectorXoshiro Catch2::Catch2)
target_include_directories(testXoshiro PRIVATE SYSTEM ${TEST_INCLUDE_DIR})
add_test(NAME testXoshiro COMMAND testXoshiro)

add_executable(testVectorXoshiro test_vector_xoshiro.cpp)
target_link_libraries(testVectorXoshiro PRIVATE vectorXoshiro Catch2::Catch2)
target_include_directories(testVectorXoshiro PRIVATE SYSTEM ${TEST_INCLUDE_DIR})
add_test(NAME testVectorXoshiro COMMAND testVectorXoshiro)


add_executable(benchmarks benchmarks.cpp)
target_link_libraries(benchmarks PRIVATE vectorXoshiro nanobench)
target_compile_options(benchmarks PRIVATE -fwhole-program)
set_property(TARGET benchmarks PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
