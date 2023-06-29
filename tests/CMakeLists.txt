
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
file(DOWNLOAD https://prng.di.unimi.it/xoshiro256plusplus.c ${TEST_INCLUDE_DIR}/xoshiro256plusplus.c)

add_executable(testSplitMix split_mix_tests.cpp)
target_link_libraries(testSplitMix PRIVATE vectorXoshiro Catch2::Catch2 benchmark::benchmark)
target_include_directories(testSplitMix PRIVATE SYSTEM ${TEST_INCLUDE_DIR})

add_executable(testXoshiro test_xoshiro.cpp)
target_link_libraries(testXoshiro PRIVATE vectorXoshiro Catch2::Catch2 benchmark::benchmark)
target_include_directories(testXoshiro PRIVATE SYSTEM ${TEST_INCLUDE_DIR})

add_executable(testSplitMixVector testVectorSplitMix64.cpp)
target_link_libraries(testSplitMixVector PRIVATE vectorXoshiro Catch2::Catch2 benchmark::benchmark)
target_include_directories(testSplitMixVector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(testVectorXoshiroPlusPlus testVectorXoshiroPlusPlus.cpp)
target_link_libraries(testVectorXoshiroPlusPlus PRIVATE vectorXoshiro Catch2::Catch2 benchmark::benchmark)
target_include_directories(testVectorXoshiroPlusPlus PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
