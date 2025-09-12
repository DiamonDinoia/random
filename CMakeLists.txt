cmake_minimum_required(VERSION 3.16)

project(random VERSION 0.1 LANGUAGES CXX)


# if no build type was specified, default to Release
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif ()

set(CPM_DOWNLOAD_VERSION 0.40.6)
set(XSIMD_DOWNLOAD_VERSION feat/faster-shift-rotation)

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
        GITHUB_REPOSITORY DiamonDinoia/xsimd
        GIT_TAG ${XSIMD_DOWNLOAD_VERSION}
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
)

include(CheckCXXCompilerFlag)
include(CheckLinkerFlag)
# List candidate standards in descending order.
set(CXX_STANDARDS "26" "23" "20" "17")
set(SUPPORTED_CXX_STANDARD "")

foreach (std IN LISTS CXX_STANDARDS)
    set(flag "-std=c++${std}")
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


function(add_compile_flags_if_supported OUTVAR)
    foreach (flag ${ARGN})
        string(MAKE_C_IDENTIFIER "HAS_${flag}" var)   # e.g. -fno-plt -> HAS__FNO_PLT
        check_cxx_compiler_flag("${flag}" ${var})
        if (${var})
            list(APPEND ${OUTVAR} "${flag}")
        endif ()
    endforeach ()
    set(${OUTVAR} "${${OUTVAR}}" PARENT_SCOPE)
endfunction()

function(add_link_flags_if_supported OUTVAR)
    foreach (flag ${ARGN})
        string(MAKE_C_IDENTIFIER "HAS_LNK_${flag}" var)
        check_linker_flag(CXX "${flag}" ${var})
        if (${var})
            list(APPEND ${OUTVAR} "${flag}")
        endif ()
    endforeach ()
    set(${OUTVAR} "${${OUTVAR}}" PARENT_SCOPE)
endfunction()

# --- choose flags (no PGO, no -march) ---
set(PERF_COMPILE_FLAGS "")
set(PERF_LINK_FLAGS "")

# LTO: prefer thin/full if available, else plain
check_cxx_compiler_flag("-flto=thin" HAS_LTO_THIN)
check_cxx_compiler_flag("-flto=full" HAS_LTO_FULL)
check_cxx_compiler_flag("-flto" HAS_LTO)

if (HAS_LTO_THIN)
    list(APPEND PERF_COMPILE_FLAGS -flto=thin)
elseif (HAS_LTO_FULL)
    list(APPEND PERF_COMPILE_FLAGS -flto=full)
elseif (HAS_LTO)
    list(APPEND PERF_COMPILE_FLAGS -flto)
endif ()

# MSVC compile-side LTCG (harmless elsewhere because of the check)
add_compile_flags_if_supported(PERF_COMPILE_FLAGS /GL)

# Devirtualization helpers + DCE/ICF enablers (compiler side)
add_compile_flags_if_supported(PERF_COMPILE_FLAGS
        -fno-semantic-interposition
        -fwhole-program-vtables
        -fstrict-vtable-pointers
        -fipa-pta
        -fdevirtualize-at-ltrans
        -ffunction-sections -fdata-sections
        -fvisibility-inlines-hidden
        -funroll-loops
        -fpeel-loops
        -funswitch-loops
        -Ofast
        -fimplicit-constexpr
)

# ELF niceties
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_compile_flags_if_supported(PERF_COMPILE_FLAGS -fno-plt -fuse-ld=lld)
endif ()

# Link-time: make GC/ICF effective, LTO at link, platform trims
add_link_flags_if_supported(PERF_LINK_FLAGS
        -Wl,--gc-sections
        -Wl,--icf=all
        -Wl,-dead_strip   # macOS ld64
        /LTCG /OPT:ICF /OPT:REF /INCREMENTAL:NO  # MSVC
)

# Mirror LTO to the linker (pick the same style as compile)
if (HAS_LTO_THIN)
    add_link_flags_if_supported(PERF_LINK_FLAGS -flto=thin)
elseif (HAS_LTO_FULL)
    add_link_flags_if_supported(PERF_LINK_FLAGS -flto=full)
elseif (HAS_LTO)
    add_link_flags_if_supported(PERF_LINK_FLAGS -flto)
endif ()

# --- wire into your existing lists (only for optimized configs) ---
list(APPEND COMPILE_OPTIONS
        $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:${PERF_COMPILE_FLAGS}>
)

set(LINK_OPTIONS "")
list(APPEND LINK_OPTIONS
        $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:${PERF_LINK_FLAGS}>
)

add_library(random STATIC src/xoshiro_simd.cpp)
target_include_directories(random PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(random PUBLIC xsimd)
set_target_properties(random PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(random PRIVATE ${COMPILE_OPTIONS})
target_link_options(random PRIVATE ${LINK_OPTIONS})
target_compile_features(random PRIVATE cxx_std_${SUPPORTED_CXX_STANDARD})

foreach (MARCH_VERSION IN ITEMS x86-64 x86-64-v2 x86-64-v3 x86-64-v4)
    # hard-coded nearest valid -mtune for each x86-64 level
    if (MARCH_VERSION STREQUAL "x86-64")
        set(MTUNE "generic")
    elseif (MARCH_VERSION STREQUAL "x86-64-v2")
        set(MTUNE "sandybridge")
    elseif (MARCH_VERSION STREQUAL "x86-64-v3")
        set(MTUNE "haswell")
    elseif (MARCH_VERSION STREQUAL "x86-64-v4")
        set(MTUNE "skylake-avx512")  # falls back gracefully if ignored by Clang
    endif ()

    string(REPLACE "-" "_" TARGET_SUFFIX "${MARCH_VERSION}")
    set(SIMD_SOURCE_TARGET "xoshiro_simd_source_${TARGET_SUFFIX}")

    add_library(${SIMD_SOURCE_TARGET} OBJECT src/xoshiro_simd_dispatch.cpp)
    target_include_directories(${SIMD_SOURCE_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(${SIMD_SOURCE_TARGET} PRIVATE xsimd)
    target_compile_options(${SIMD_SOURCE_TARGET} PRIVATE ${COMPILE_OPTIONS} -march=${MARCH_VERSION} -mtune=${MTUNE})
    set_target_properties(${SIMD_SOURCE_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_options(${SIMD_SOURCE_TARGET} PRIVATE ${LINK_OPTIONS})
    target_compile_features(${SIMD_SOURCE_TARGET} PRIVATE cxx_std_${SUPPORTED_CXX_STANDARD})
    target_sources(random PRIVATE $<TARGET_OBJECTS:${SIMD_SOURCE_TARGET}>)
endforeach ()


if (ENABLE_TESTS)
    add_subdirectory(tests)
endif ()


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
