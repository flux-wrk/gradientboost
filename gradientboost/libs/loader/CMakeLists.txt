aux_source_directory(. loader_SRC)

add_library(loader SHARED
    ${loader_SRC})

target_link_libraries(loader
    m)
