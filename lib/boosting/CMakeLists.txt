include_directories(${CMAKE_SOURCE_DIR})

aux_source_directory(. boosting_SRC)

add_library(boosting STATIC
    ${boosting_SRC})

target_link_libraries(boosting
    m)
