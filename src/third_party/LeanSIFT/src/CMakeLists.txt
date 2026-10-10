add_library(lsift
  executor.cpp
  kernels.cpp
  sift.cpp)
add_library(LeanSift::lsift ALIAS lsift)

include(GenerateExportHeader)
generate_export_header(lsift
  BASE_NAME LSIFT
  EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/lsift/export.hpp)

set_target_properties(lsift PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})
if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(lsift PUBLIC LSIFT_STATIC_DEFINE)
endif()

target_compile_features(lsift PUBLIC cxx_std_17)
target_include_directories(lsift
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lsift PRIVATE hwy::hwy Threads::Threads)

option(LSIFT_PROFILE "Print the time of each pipeline phase to stderr (development aid)" OFF)
if(LSIFT_PROFILE)
  target_compile_definitions(lsift PRIVATE LSIFT_PROFILE)
endif()

if(MSVC)
  target_compile_options(lsift PRIVATE /W4 /wd4324 /permissive- /Zc:__cplusplus /utf-8)
else()
  target_compile_options(lsift PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(LSIFT_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  install(TARGETS lsift EXPORT LeanSiftTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/lsift DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(FILES ${PROJECT_BINARY_DIR}/include/lsift/export.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lsift)
  install(EXPORT LeanSiftTargets
    NAMESPACE LeanSift::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LeanSift)
  configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/LeanSiftConfig.cmake.in
    ${PROJECT_BINARY_DIR}/LeanSiftConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LeanSift)
  write_basic_package_version_file(${PROJECT_BINARY_DIR}/LeanSiftConfigVersion.cmake
    COMPATIBILITY SameMinorVersion)
  install(FILES
    ${PROJECT_BINARY_DIR}/LeanSiftConfig.cmake
    ${PROJECT_BINARY_DIR}/LeanSiftConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LeanSift)
endif()
