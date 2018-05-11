include_directories(${CMAKE_SOURCE_DIR})

add_executable(gradientboost
    ${pir_ml_headers}
    main.cpp)

target_link_libraries(gradientboost
    loader
    preprocessing
    boosting
    tbb CLI11
        )
