aux_source_directory(. loader_SRC)

add_library(loader SHARED
    ${loader_SRC} ../desicion_tree/DataFrame.h)

target_link_libraries(loader
    m)
