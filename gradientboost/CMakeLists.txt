include_directories(${CMAKE_SOURCE_DIR})

add_executable(gradientboost main.cpp)

target_link_libraries(gradientboost
        loader
        preprocessing
        boosting
        tbb
        CLI11
        )
