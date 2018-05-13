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

namespace NGradientBoost {

    class DecisionTree {
     public:
        size_t depth_;
        std::vector<size_t> splitting_features_;

        std::vector<float_t> leaf_answers_{};

        explicit DecisionTree(size_t depth);
        explicit DecisionTree(std::istream& stream);

        void Save(std::ostream& stream) const;
        std::vector<float_t> Predict(const std::vector<std::vector<float_t>>& data) const;
    };

}
