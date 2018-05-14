#pragma once
#include <vector>
#include <string>
#include <set>
#include <limits>
#include <fstream>
#include <iostream>
#include <cfloat>
#include <cmath>

#include "lib/preprocessing/DataFrame.h"
#include "tbb/mutex.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

namespace NGradientBoost {

    class DecisionTree {
     public:
        size_t depth_;
        std::vector<size_t> splitting_features_;
        std::vector<float_t> leaf_results_{};

        explicit DecisionTree(size_t depth);
        explicit DecisionTree(std::istream& stream);

        void Save(std::ostream& stream) const;
        std::vector<float_t> Predict(const std::vector<std::vector<float_t>>& data) const;
        void Fit(const DataFrame& dataframe,
                 const Target& target,
                 const Target& baseline_predictions,
                 Target& temp_predictions);

    };

} // namespace NGradientBoost
