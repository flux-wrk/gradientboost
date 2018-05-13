add_executable(gradientboost main.cpp)

target_include_directories(gradientboost PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(gradientboost
        loader
        preprocessing
        boosting
        tbb
        decision_tree
        CLI11
        )
