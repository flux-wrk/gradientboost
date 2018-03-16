cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-march=native -flto -O3")
endif()

include_directories(.)

file(GLOB headers
        "*.hpp"
        "*.h"
)

add_executable(gradientboost ${pir_ml_headers} main.cpp)
target_link_libraries(gradientboost tbb)
