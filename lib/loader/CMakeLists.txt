aux_source_directory(. loader_SRC)

add_library(loader STATIC
    ${loader_SRC})

target_link_libraries(loader
        tbb
    m)
