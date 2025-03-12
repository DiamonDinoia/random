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


if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_options(-fstack-protector-all)
    add_compile_options(-Wall -Wextra -pedantic -fno-omit-frame-pointer -g -O0)

endif ()
if (CMAKE_BUILD_TYPE MATCHES Release)
    add_compile_options(-O3)
endif ()

add_library(vectorXoshiro INTERFACE)
target_include_directories(vectorXoshiro INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(vectorXoshiro INTERFACE cxx_std_20) # This means at least C++17
target_link_libraries(vectorXoshiro INTERFACE xsimd)
#target_compile_options(vectorXoshiro INTERFACE -fimplicit-constexpr)
if (ENABLE_TESTS)
    add_subdirectory(tests)
endif ()
