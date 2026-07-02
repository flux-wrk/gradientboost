add_library(loader STATIC
        CSV.h
        CSV.cpp
    )

target_include_directories(loader PUBLIC ${CMAKE_SOURCE_DIR})
