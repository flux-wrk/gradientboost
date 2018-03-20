aux_source_directory(. desicion_tree_SRC)

add_library(desicion_tree STATIC
    ${desicion_tree_SRC})

target_link_libraries(desicion_tree
    m)
