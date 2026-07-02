add_library(decision_tree STATIC
        Tree.h
        Tree.cpp)

target_include_directories(decision_tree PUBLIC ${CMAKE_SOURCE_DIR})
