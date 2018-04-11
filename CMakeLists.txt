cmake_minimum_required(VERSION 3.5)

# set(CMAKE_BUILD_TYPE Debug)

project(gradientboost)

include(CMakeConfig.txt)

# 3D-Party
add_subdirectory(3d-party)

# library itself
add_subdirectory(gradientboost)
add_subdirectory(lib)
add_subdirectory(tests)
