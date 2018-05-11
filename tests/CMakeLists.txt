set(target gradientboost_test)

include_directories(${CMAKE_SOURCE_DIR})

aux_source_directory(. gradientboost_test_SRC)

add_executable(${target}
    ${pir_ml_headers}
    ${gradientboost_test_SRC})

target_include_directories(gmock_main SYSTEM BEFORE INTERFACE
    "${gtest_SOURCE_DIR}/include"
    "${gmock_SOURCE_DIR}/include")

target_link_libraries(${target}
    loader
    decision_tree
    tbb
    gmock_main)

add_test(NAME ${target} COMMAND ${target})
