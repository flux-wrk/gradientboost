#include <vector>
#include <string>
#include <set>
#include <limits>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cassert>

#include "lib/decision_tree/Tree.h"
#include "lib/preprocessing/DataFrame.h"
#include "tbb/mutex.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

namespace NGradientBoost {

    class BoostedClassifier {
     public:
        BoostedClassifier(size_t tree_count, size_t depth, float_t learning_rate) {
            this->learning_rate_ = learning_rate;
            this->tree_count_ = tree_count;
            this->tree_depth_ = depth;
        }
        explicit BoostedClassifier(std::istream& stream);

        BoostedClassifier& Fit(const Dataset& data, const Target& target);
        float_t Eval(const Dataset& data, const Target& target) const;
        Target Predict(const Dataset& data) const;

        bool Save(std::ostream& stream) const;

     private:
        static float_t MSE(const Target& predicted, const Target& actual);

        std::vector<DecisionTree> trees_;
        float_t learning_rate_{};
        size_t tree_count_;
        size_t tree_depth_;
    };

} // namespace NGradientBoost
