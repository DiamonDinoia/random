cmake_minimum_required(VERSION 3.16)

project(random VERSION 0.1 LANGUAGES CXX)

set(CPM_DOWNLOAD_VERSION 0.40.6)
set(XSIMD_DOWNLOAD_VERSION 13.2.0)

option(ENABLE_TESTS "Enable tests" ON)
option(ENABLE_PYTHON "Enable Python bindings" ON)


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
#
#set(COMPILE_OPTIONS
#        $<$<CONFIG:Debug>:
#        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fstack-protector-all -Wall -Wextra -pedantic -fno-omit-frame-pointer -g -O0>
#        $<$<CXX_COMPILER_ID:MSVC>:/W4 /Od /Zi>
#        >
#        $<$<CONFIG:Release>:
#        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3>
#        $<$<CXX_COMPILER_ID:MSVC>:/O2>
#        >
#        $<$<CONFIG:RelWithDebInfo>:
#        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2 -g>
#        $<$<CXX_COMPILER_ID:MSVC>:/O2 /Zi>
#        >
#        $<$<CONFIG:MinSizeRel>:
#        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Os>
#        $<$<CXX_COMPILER_ID:MSVC>:/O1>
#        >
#)

add_library(random STATIC src/xoshiro_simd.cpp)
target_include_directories(random PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(random PUBLIC xsimd)
target_compile_options(random PRIVATE ${COMPILE_OPTIONS})
set_target_properties(random PROPERTIES POSITION_INDEPENDENT_CODE ON)

foreach (MARCH_VERSION IN ITEMS x86-64 x86-64-v2 x86-64-v3 x86-64-v4)
    string(REPLACE "-" "_" TARGET_SUFFIX "${MARCH_VERSION}")
    set(SIMD_SOURCE_TARGET "xoshiro_simd_source_${TARGET_SUFFIX}")
    add_library(${SIMD_SOURCE_TARGET} OBJECT src/xoshiro_simd_dispatch.cpp)
    target_include_directories(${SIMD_SOURCE_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(${SIMD_SOURCE_TARGET} PRIVATE ${COMPILE_OPTIONS} xsimd)
    target_compile_options(${SIMD_SOURCE_TARGET} PRIVATE -march=${MARCH_VERSION})
    set_target_properties(${SIMD_SOURCE_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_sources(random PRIVATE $<TARGET_OBJECTS:${SIMD_SOURCE_TARGET}>)
endforeach ()


if (ENABLE_TESTS)
    add_subdirectory(tests)
endif ()

include(CheckCXXCompilerFlag)

# List candidate standards in descending order.
set(CXX_STANDARDS "23" "20" "17")
set(SUPPORTED_CXX_STANDARD "")

foreach (std IN LISTS CXX_STANDARDS)
    if (std STREQUAL "23")
        set(flag "-std=c++23")
    else ()
        set(flag "-std=c++${std}")
    endif ()
    # Create a unique variable name for each check.
    string(CONCAT var "COMPILER_SUPPORTS_CXX" std)
    check_cxx_compiler_flag(${flag} ${var})
    if (${var})
        set(SUPPORTED_CXX_STANDARD ${std})
        break()  # Use the highest supported standard.
    endif ()
endforeach ()

if (NOT SUPPORTED_CXX_STANDARD)
    message(FATAL_ERROR "The compiler does not support C++17 or higher.")
endif ()

target_compile_features(random PRIVATE cxx_std_${SUPPORTED_CXX_STANDARD})

if (ENABLE_PYTHON)
    add_subdirectory(python)
else ()
    # Installation rules
    install(TARGETS random
            EXPORT randomTargets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
    )

    install(DIRECTORY include/ DESTINATION include)

    install(EXPORT randomTargets
            FILE randomTargets.cmake
            NAMESPACE prng::
            DESTINATION lib/cmake/random
    )

    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
            "${CMAKE_CURRENT_BINARY_DIR}/randomConfigVersion.cmake"
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY AnyNewerVersion
    )

    configure_file(cmake/randomConfig.cmake.in
            "${CMAKE_CURRENT_BINARY_DIR}/randomConfig.cmake"
            @ONLY
    )

    install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/randomConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/randomConfigVersion.cmake"
            DESTINATION lib/cmake/random
    )

endif ()
