add_subdirectory(libs)

# include_directories(libs/decision_tree)
include_directories(libs/loader)

add_executable(gradientboost
    ${pir_ml_headers}
    main.cpp)

target_link_libraries(gradientboost
    loader
    tbb)
