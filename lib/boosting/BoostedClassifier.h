#include <vector>
#include <string>
#include <set>
#include <limits>
#include <fstream>
#include <iostream>

#include "lib/preprocessing/DataFrame.h"
#include "tbb/mutex.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/mutex.h"
#include "tbb/task_scheduler_init.h"

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
    float_t Eval(const Dataset& data, const Target& target);
    Target Predict(const Dataset& data) const;

    bool Save(std::ostream& stream) const;
private:

    class DecisionTree {
    public:
        size_t depth_;
        std::vector<size_t> splitting_features_;
        std::vector<float_t> leaf_answers_{};

        explicit DecisionTree(size_t depth);

        std::vector<float_t> Predict(const std::vector<std::vector<float_t>>& data) const;
    };

    static float_t MSE(const Target& predicted, const Target& actual);

    std::vector<DecisionTree> trees_;
    float_t learning_rate_{};
    size_t tree_count_;
    size_t tree_depth_;
//    bool valid_{false};
};

} // namespace NGradientBoost
