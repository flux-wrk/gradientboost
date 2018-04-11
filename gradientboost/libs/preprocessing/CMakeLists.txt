aux_source_directory(. preprocessing_SRC)

add_library(preprocessing STATIC
    ${preprocessing_SRC})

target_link_libraries(preprocessing
    m)
