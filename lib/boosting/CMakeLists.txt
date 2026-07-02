add_library(boosting STATIC
        BoostedClassifier.cpp
        BoostedClassifier.h
        )

target_include_directories(boosting PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(boosting
        PUBLIC decision_tree
        PRIVATE m)
