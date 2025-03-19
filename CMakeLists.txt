cmake_minimum_required(VERSION 3.16)

project(XOSHIRO)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(DCMAKE_RULE_MESSAGES OFF)

# version of the libraries to be used
set(CPM_DOWNLOAD_VERSION 0.40.6)
set(XSIMD_DOWNLOAD_VERSION 13.2.0)

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


CPMAddPackage(
        NAME xsimd
        GITHUB_REPOSITORY QuantStack/xsimd
        GIT_TAG ${XSIMD_DOWNLOAD_VERSION}
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
)

set(COMPILE_OPTIONS
        $<$<CONFIG:Debug>:
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fstack-protector-all -Wall -Wextra -pedantic -fno-omit-frame-pointer -g -O0>
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /Od /Zi>
        >
        $<$<CONFIG:Release>:
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>
        >
        $<$<CONFIG:RelWithDebInfo>:
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2 -g>
        $<$<CXX_COMPILER_ID:MSVC>:/O2 /Zi>
        >
        $<$<CONFIG:MinSizeRel>:
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Os>
        $<$<CXX_COMPILER_ID:MSVC>:/O1>
        >
)

add_library(vectorXoshiro STATIC src/xoshiro.cpp)
target_include_directories(vectorXoshiro PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(vectorXoshiro PRIVATE cxx_std_17) # This means at least C++17
target_link_libraries(vectorXoshiro PUBLIC xsimd)
target_compile_options(vectorXoshiro PRIVATE ${COMPILE_OPTIONS})


foreach (MARCH_VERSION IN ITEMS x86-64 x86-64-v2 x86-64-v3 x86-64-v4)
    string(REPLACE "-" "_" TARGET_SUFFIX "${MARCH_VERSION}")
    set(SIMD_SOURCE_TARGET "vector_xoshiro_simd_source_${TARGET_SUFFIX}")
    add_library(${SIMD_SOURCE_TARGET} OBJECT src/xoshiro_simd.cpp)
    target_include_directories(${SIMD_SOURCE_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${SIMD_SOURCE_TARGET} PRIVATE ${COMPILE_OPTIONS} xsimd)
    target_compile_options(${SIMD_SOURCE_TARGET} PRIVATE -march=${MARCH_VERSION})
    target_sources(vectorXoshiro PRIVATE $<TARGET_OBJECTS:${SIMD_SOURCE_TARGET}>)
endforeach ()


if (ENABLE_TESTS)
    add_subdirectory(tests)
endif ()
