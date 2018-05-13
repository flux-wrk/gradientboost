#include "BoostedClassifier.h"

namespace NGradientBoost {

    BoostedClassifier& BoostedClassifier::Fit(const Dataset& data, const Target& target) {
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        auto dataframe = DataFrame(data);
        trees_.clear();
        std::vector<float_t> current_predictions(dataframe.size(), 0), temp_pred(dataframe.size(), 0);

        for (size_t iteration = 0; iteration < tree_count_; ++iteration) {

            DecisionTree weak_classifier(tree_depth_);
            weak_classifier.Fit(dataframe, target, current_predictions, temp_pred);

            current_predictions = temp_pred;
            float_t loss = BoostedClassifier::MSE(target, temp_pred);
            std::cout << "Loss on iteration " << (iteration + 1) << " : " << loss << std::endl;

            trees_.push_back(weak_classifier);
        }
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.f;
        std::cout << "\nFitting complete, time elapsed: " << duration << " seconds (" << duration / tree_count_ << " per tree)\n";

        return *this;
    }

    std::vector<float_t> BoostedClassifier::Predict(const std::vector<std::vector<float_t>>& data) const {
        auto dataframe = DataFrame(data);
        std::vector<float_t> predictions(dataframe.size());
        for (const DecisionTree& weak_clf : trees_) {
            std::vector<float_t> predictions_for_tree = weak_clf.Predict(data);
            for (size_t i = 0; i < predictions.size(); ++i) {
                predictions[i] += predictions_for_tree[i];
            }
        }
        return predictions;
    }


    bool BoostedClassifier::Save(std::ostream& stream) const {
        stream << tree_depth_ << " " << tree_count_ << " " << learning_rate_ << std::endl;
        for (const auto& tree : trees_) {
            tree.Save(stream);
        }
        return true;
    }

    BoostedClassifier::BoostedClassifier(std::istream& stream) {
        stream >> tree_depth_ >> tree_count_ >> learning_rate_;
        for (size_t idx = 0; idx < tree_count_; ++idx) {
            trees_.emplace_back(stream);
        }
    }

    float_t BoostedClassifier::Eval(const Dataset& data, const Target& target) const {
        return BoostedClassifier::MSE(Predict(data), target);
    }

    inline float_t sqr(float_t x) { return x * x; }

    float_t BoostedClassifier::MSE(const Target& predicted, const Target& actual) {
        assert(predicted.size() == actual.size());
        float_t MSE = 0.0;
        for (size_t idx = 0; idx < predicted.size(); ++idx) {
            MSE += sqr(predicted[idx] - actual[idx]);
        }
        return MSE / predicted.size();
    }

} // namespace NGradientBoost

