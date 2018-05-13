#pragma once

#include "lib/preprocessing/DataFrame.h"

#include <vector>
#include <cstdint>

namespace NGradientBoost {

    class DecisionTreeClassifier {
     public:
        explicit DecisionTreeClassifier(int leaf_count)
            : leaf_count_(leaf_count) {}
        virtual ~DecisionTreeClassifier() = default;

        virtual void Fit(const Dataset& dataset, const std::vector<Label>& labels) = 0;
        virtual std::vector<Label> Predict(const Dataset& dataset) const = 0;

     protected:
        int leaf_count_;
    };

}  // namespace NGradientBoost
