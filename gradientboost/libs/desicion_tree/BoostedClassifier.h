#include <vector>
#include <string>
#include <set>
#include <limits>
#include "DataFrame.h"

namespace NGradientBoost {

    class BoostedClassifier {
    private:
        class DecisionTree {
        public:
            size_t depth_;
            std::vector<size_t> splitting_features_;
            std::vector<float_t> leaf_answers_{};

            explicit DecisionTree(size_t depth);

            std::vector<float_t> Predict(const std::vector<std::vector<float_t>>& data) const;
        };

        std::vector<DecisionTree> trees_;
        float_t learning_rate_{};
        size_t tree_count_;
        size_t tree_depth_;

    public:
        BoostedClassifier(float_t learning_rate, size_t depth, size_t tree_count) {
            this->learning_rate_ = learning_rate;
            this->tree_count_ = tree_count;
            this->tree_depth_ = depth;
        }

        BoostedClassifier &Fit(const std::vector<std::vector<float_t>>& data, const std::vector<float_t> target);

        std::vector<float_t> Predict(const std::vector<std::vector<float_t>>& data) const;

    };
} // namespace NGradientBoost