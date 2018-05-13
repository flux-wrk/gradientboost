include_directories(${CMAKE_SOURCE_DIR})

add_library(decision_tree STATIC
        Tree.h
        Tree.cpp)

target_link_libraries(decision_tree
        tbb
    )
