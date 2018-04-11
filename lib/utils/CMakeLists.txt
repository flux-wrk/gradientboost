include_directories(${CMAKE_SOURCE_DIR})

aux_source_directory(. utils_SRC)

add_library(utils STATIC
    ${utils_SRC})

target_link_libraries(utils
    m)
