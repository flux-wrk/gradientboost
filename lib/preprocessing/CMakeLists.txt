add_library(preprocessing STATIC
        DataFrame.h
        Preprocessor.cpp
        Preprocessor.h)

target_include_directories(preprocessing PUBLIC ${CMAKE_SOURCE_DIR})
