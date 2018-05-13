include_directories(${CMAKE_SOURCE_DIR})

add_library(preprocessing STATIC
        DataFrame.h
        Preprocessor.cpp
        Preprocessor.h)

target_link_libraries(preprocessing tbb)
