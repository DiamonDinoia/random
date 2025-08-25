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

# cmake
set(COMPILE_OPTIONS
        # Common warnings and sane defaults
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wformat=2 -Wformat-security -fvisibility=hidden>
        $<$<CXX_COMPILER_ID:AppleClang>:-Wall -Wextra -Wpedantic -Wformat=2 -Wformat-security -fvisibility=hidden>
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive- /EHsc /Zc:__cplusplus /Zc:throwingNew /utf-8 /MP>

        # Debug: full debug info, frame pointers for profiling, strong stack protector,
        # and extra libstdc++ assertions on GNU/Clang.
        $<$<CONFIG:Debug>:
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O0 -g3 -fno-omit-frame-pointer -fstack-protector-strong -D_GLIBCXX_ASSERTIONS>
        $<$<CXX_COMPILER_ID:AppleClang>:-O0 -g3 -fno-omit-frame-pointer -fstack-protector-strong>
        $<$<CXX_COMPILER_ID:MSVC>:/Od /Zi /RTC1>
        >

        # RelWithDebInfo: optimized with debug info, keep frame pointers, harden, and disable asserts.
        $<$<CONFIG:RelWithDebInfo>:
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -g -fno-omit-frame-pointer -fstack-protector-strong -DNDEBUG -D_FORTIFY_SOURCE=2>
        $<$<CXX_COMPILER_ID:AppleClang>:-O2 -g -fno-omit-frame-pointer -fstack-protector-strong -DNDEBUG>
        $<$<CXX_COMPILER_ID:MSVC>:/O2 /Zi /Gy /DNDEBUG>
        >

        # Release: maximum performance and harden.
        $<$<CONFIG:Release>:
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -DNDEBUG -D_FORTIFY_SOURCE=2>
        $<$<CXX_COMPILER_ID:AppleClang>:-O3 -DNDEBUG>
        $<$<CXX_COMPILER_ID:MSVC>:/O2 /Gy /DNDEBUG>
        >

        # MinSizeRel: prioritize size, still harden.
        $<$<CONFIG:MinSizeRel>:
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Os -DNDEBUG -D_FORTIFY_SOURCE=2>
        $<$<CXX_COMPILER_ID:AppleClang>:-Oz -DNDEBUG>
        $<$<CXX_COMPILER_ID:MSVC>:/O1 /Gy /DNDEBUG>
        >
)
# 1) Detect support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-flto=full" COMPILER_SUPPORTS_LTO_FULL)
check_cxx_compiler_flag("-flto" COMPILER_SUPPORTS_LTO_PLAIN)
check_cxx_compiler_flag("-flto=thin" COMPILER_SUPPORTS_LTO_THIN)
check_cxx_compiler_flag("-fimplicit-constexpr" COMPILER_SUPPORTS_FIMPLICIT_CONSTEXPR)

# Devirtualization-related checks
check_cxx_compiler_flag("-fno-semantic-interposition" COMPILER_SUPPORTS_NO_SEMINT)
check_cxx_compiler_flag("-fwhole-program-vtables" COMPILER_SUPPORTS_WPV)
check_cxx_compiler_flag("-fstrict-vtable-pointers" COMPILER_SUPPORTS_STRICT_VTP)
check_cxx_compiler_flag("-fipa-pta" COMPILER_SUPPORTS_IPA_PTA)
check_cxx_compiler_flag("-fdevirtualize-at-ltrans" COMPILER_SUPPORTS_DEVIRT_LTRANS)

# 2) Append conditionally to your existing COMPILE_OPTIONS
list(APPEND COMPILE_OPTIONS
        # Already present in your snippet:
        $<$<BOOL:${COMPILER_SUPPORTS_FIMPLICIT_CONSTEXPR}>:-fimplicit-constexpr>

        # Prefer ThinLTO on Clang/AppleClang when available; fall back to full.
        $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,$<CXX_COMPILER_ID:Clang,AppleClang>,$<BOOL:${COMPILER_SUPPORTS_LTO_THIN}>>:-flto=thin>
        $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,$<CXX_COMPILER_ID:Clang,AppleClang>,$<NOT:$<BOOL:${COMPILER_SUPPORTS_LTO_THIN}>>,$<BOOL:${COMPILER_SUPPORTS_LTO_FULL}>>:-flto=full>

        # GCC LTO
        $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,$<CXX_COMPILER_ID:GNU>,$<BOOL:${COMPILER_SUPPORTS_LTO_PLAIN}>>:-flto>

        # MSVC LTCG (compile side)
        $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,$<CXX_COMPILER_ID:MSVC>>:/GL>

        # Help devirtualization by disabling interposition (GCC/Clang/AppleClang)
        $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,$<BOOL:${COMPILER_SUPPORTS_NO_SEMINT}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-fno-semantic-interposition>

        # Clang: whole-program vtables + stricter vtable semantics
        $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,$<CXX_COMPILER_ID:Clang,AppleClang>,$<BOOL:${COMPILER_SUPPORTS_WPV}>>:-fwhole-program-vtables>
        $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,$<CXX_COMPILER_ID:Clang,AppleClang>,$<BOOL:${COMPILER_SUPPORTS_STRICT_VTP}>>:-fstrict-vtable-pointers>

        # GCC: stronger IPA points-to + devirt at LTRANS
        $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,$<CXX_COMPILER_ID:GNU>,$<BOOL:${COMPILER_SUPPORTS_IPA_PTA}>>:-fipa-pta>
        $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,$<CXX_COMPILER_ID:GNU>,$<BOOL:${COMPILER_SUPPORTS_DEVIRT_LTRANS}>>:-fdevirtualize-at-ltrans>

        # Complements -fvisibility=hidden for template inlines (harmless on non-shared too)
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fvisibility-inlines-hidden>
)

# 3) Linker side (so LTO actually applies)
#    If you’re already collecting link flags elsewhere, merge these similarly.
set(LINK_OPTIONS "")
list(APPEND LINK_OPTIONS
        # Clang/AppleClang: ThinLTO preferred
        $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,$<CXX_COMPILER_ID:Clang,AppleClang>,$<BOOL:${COMPILER_SUPPORTS_LTO_THIN}>>:-flto=thin>
        $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,$<CXX_COMPILER_ID:Clang,AppleClang>,$<NOT:$<BOOL:${COMPILER_SUPPORTS_LTO_THIN}>>,$<BOOL:${COMPILER_SUPPORTS_LTO_FULL}>>:-flto=full>

        # GCC: plain LTO
        $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,$<CXX_COMPILER_ID:GNU>,$<BOOL:${COMPILER_SUPPORTS_LTO_PLAIN}>>:-flto>

        # MSVC: LTCG at link + disable incremental to avoid conflicts
        $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,$<CXX_COMPILER_ID:MSVC>>:/LTCG /INCREMENTAL:NO>
)

# Apply (example)
# target_compile_options(your_target PRIVATE ${COMPILE_OPTIONS})
# target_link_options(your_target PRIVATE ${LINK_OPTIONS})


add_library(random STATIC src/xoshiro_simd.cpp)
target_include_directories(random PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(random PUBLIC xsimd)
set_target_properties(random PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(random PRIVATE ${COMPILE_OPTIONS})
target_link_options(random PRIVATE ${LINK_OPTIONS})

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
