add_executable(gradientboost main.cpp)

target_link_libraries(gradientboost
        PRIVATE loader preprocessing boosting decision_tree CLI11::CLI11
        )
