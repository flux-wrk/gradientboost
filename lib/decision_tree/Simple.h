#pragma once

#include "Classifier.h"
#include "Tree.h"

#include "lib/preprocessing/DataFrame.h"

#include <vector>

namespace NGradientBoost {

class SimpleDecisionTreeClassifier : public DecisionTreeClassifier {
public:
    explicit SimpleDecisionTreeClassifier(int leaf_count) :
        DecisionTreeClassifier(leaf_count),
        tree_(leaf_count)
    { }

    void Fit(const Dataset& dataset, const std::vector<Label>& labels) override;
    std::vector<Label> Predict(const Dataset& dataset) const override;

private:
    Tree tree_;
};

} // NGradientBoost

