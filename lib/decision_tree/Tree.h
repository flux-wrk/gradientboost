#pragma once

#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <set>
#include <vector>

#include "lib/preprocessing/DataFrame.h"

namespace NGradientBoost {

class DecisionTree {
public:
    size_t depth_;
    std::vector<size_t> splitting_features_;
    std::vector<float_t> leaf_results_{};

    explicit DecisionTree(size_t depth);
    explicit DecisionTree(std::istream& stream);

    void Save(std::ostream& stream) const;
    std::vector<float_t> Predict(const DataFrame& dataframe) const;
    void Fit(const DataFrame& dataframe, const Target& target, const Target& baseline_predictions,
             Target& temp_predictions);
};

} // namespace NGradientBoost
