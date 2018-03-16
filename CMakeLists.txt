cmake_minimum_required (VERSION 3.5)
project (gradientboost)

include(CMakeConfig.txt)

# 3D-Party
add_subdirectory(3d-party)

# library itself 
add_subdirectory (${CMAKE_CURRENT_LIST_DIR}/gradientboost)

string(REPLACE " -Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
