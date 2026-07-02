cmake_minimum_required(VERSION 4.0)

project(gradientboost LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compile options
add_compile_options(-Wall -Wextra -Werror -Wno-unused-parameter -Wno-missing-field-initializers)

# Release optimizations
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# GCC-specific LTO
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-march=native -flto)
endif()

# Fetch CLI11 v2.6.2
include(FetchContent)
FetchContent_Declare(
    cli11_proj
    QUIET
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.6.2
)
FetchContent_MakeAvailable(cli11_proj)

# Library modules
add_subdirectory(lib)

# Executable
add_subdirectory(gradientboost)
