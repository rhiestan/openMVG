# Smoke test of the installed package: a project outside the source tree
# that finds openMVG with find_package and links its libraries.
cmake_minimum_required(VERSION 3.16)
project(openMVG_downstream CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMVG REQUIRED)

add_executable(downstream main.cpp)
target_link_libraries(downstream PRIVATE
  OpenMVG::openMVG_sfm
  OpenMVG::openMVG_sift)
