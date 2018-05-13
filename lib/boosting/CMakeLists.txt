include_directories(${CMAKE_SOURCE_DIR})

add_library(boosting STATIC
        BoostedClassifier.cpp
        BoostedClassifier.h
        )

target_link_libraries(boosting
        tbb
        decision_tree
        m)
