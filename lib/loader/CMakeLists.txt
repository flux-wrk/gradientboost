include_directories(${CMAKE_SOURCE_DIR})

add_library(loader STATIC
        CSV.h
        CSV.cpp
    )

target_link_libraries(loader tbb)
