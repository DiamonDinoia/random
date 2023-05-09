
CPMAddPackage(
        NAME catch2
        GITHUB_REPOSITORY catchorg/Catch2
        VERSION 2.13.7
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM
)

set(TEST_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)

file(MAKE_DIRECTORY ${TEST_INCLUDE_DIR})
file(DOWNLOAD https://prng.di.unimi.it/splitmix64.c ${TEST_INCLUDE_DIR}/splitmix64.c)

add_executable(testSplitMix split_mix_tests.cpp)
target_link_libraries(testSplitMix PRIVATE vectorXoroshiro Catch2::Catch2 benchmark::benchmark)
target_include_directories(testSplitMix PRIVATE ${TEST_INCLUDE_DIR})
