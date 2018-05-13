cmake_minimum_required(VERSION 3.5)

# set(CMAKE_BUILD_TYPE Debug)

project(gradientboost)

include(CMakeConfig.txt)

# 3D-Party
set(3D_PARTY_DIR ${CMAKE_SOURCE_DIR}/3d-party)
add_subdirectory(${3D_PARTY_DIR})

# library itself
add_subdirectory(gradientboost)
add_subdirectory(lib)
