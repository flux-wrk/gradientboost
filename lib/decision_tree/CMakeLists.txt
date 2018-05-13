include_directories(${CMAKE_SOURCE_DIR})

aux_source_directory(. decision_tree_SRC)

add_library(decision_tree STATIC
    ${decision_tree_SRC})

target_link_libraries(decision_tree
        tbb
    )
