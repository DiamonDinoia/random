if (NOT SKBUILD)
    message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

find_package(Python
        REQUIRED COMPONENTS Interpreter Development.Module NumPy
        OPTIONAL_COMPONENTS Development.SABIModule)

set(NANOBIND_VERSION 2.9.2)

CPMAddPackage(
    NAME nanobind
    GITHUB_REPOSITORY wjakob/nanobind
    VERSION ${NANOBIND_VERSION}
)

nanobind_add_module(pyrandom_ext NB_STATIC STABLE_ABI FREE_THREADED LTO NOMINSIZE pyrandom.cpp)
target_link_libraries(pyrandom_ext PRIVATE random Python::NumPy)
target_compile_options(pyrandom_ext PRIVATE ${COMPILE_OPTIONS})
target_link_options(pyrandom_ext PRIVATE ${LINK_OPTIONS})
install(TARGETS pyrandom_ext LIBRARY DESTINATION .)
