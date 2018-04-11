set(target gradientboost_test)

include_directories(${CMAKE_SOURCE_DIR})

aux_source_directory(. gradientboost_test_SRC)

add_executable(${target}
    ${pir_ml_headers}
    ${gradientboost_test_SRC})

target_link_libraries(${target}
    loader
    decision_tree
    utils
    tbb
    gtest
    gmock)

add_test(NAME ${target} COMMAND ${target})
