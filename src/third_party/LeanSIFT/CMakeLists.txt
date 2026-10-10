cmake_minimum_required(VERSION 3.21)

project(LeanSift
  VERSION 0.1.0
  DESCRIPTION "Fast SIMD and multi-threaded SIFT, compatible with VLFeat"
  LANGUAGES CXX)

option(LSIFT_BUILD_TESTS "Build the LeanSift tests (needs network access for dependencies)" ${PROJECT_IS_TOP_LEVEL})
option(LSIFT_BUILD_TOOLS "Build the lsift command-line tool and benchmark" ${PROJECT_IS_TOP_LEVEL})
option(LSIFT_USE_SYSTEM_HWY "Use an installed Google Highway (find_package) instead of fetching it" OFF)
option(LSIFT_INSTALL "Generate the install target" ${PROJECT_IS_TOP_LEVEL})
set(LSIFT_VLFEAT_SOURCE_DIR "" CACHE PATH
  "VLFeat source tree used as the reference in tests and benchmarks (fetched from GitHub when empty)")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Google Highway (portable SIMD with runtime dispatch)
# ---------------------------------------------------------------------------
if(LSIFT_USE_SYSTEM_HWY)
  find_package(hwy 1.0 REQUIRED)
else()
  set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_INSTALL ${LSIFT_INSTALL} CACHE BOOL "" FORCE)
  set(HWY_FORCE_STATIC_LIBS ON CACHE BOOL "" FORCE)
  FetchContent_Declare(hwy
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 1.4.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(hwy)
  # Treat Highway's headers as system headers, so that our warning flags
  # (-Wpedantic) do not apply to them.
  get_target_property(_lsift_hwy_includes hwy INTERFACE_INCLUDE_DIRECTORIES)
  set_target_properties(hwy PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_lsift_hwy_includes}")
  if(NOT TARGET hwy::hwy)
    add_library(hwy::hwy ALIAS hwy)
  endif()
endif()

add_subdirectory(src)

if(LSIFT_BUILD_TESTS OR LSIFT_BUILD_TOOLS)
  add_subdirectory(third_party)
  add_subdirectory(support)
endif()

if(LSIFT_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

if(LSIFT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
