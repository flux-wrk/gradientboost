#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "lib/preprocessing/DataFrame.h"
#include "lib/decision_tree/Simple.h"

using namespace NGradientBoost;

TEST(SimpleDecisionTree, BaseTest) {
    const auto dataset = Dataset{
        { 1, 0, 0 },
        { 2, 0, 0 },
    };
    const auto labels = std::vector<Label>{0, 1};

    SimpleDecisionTreeClassifier decisionTree(5);
    decisionTree.Fit(dataset, labels);
    const auto predict = decisionTree.Predict(dataset);

    EXPECT_THAT(predict, ::testing::ContainerEq(labels));
}


