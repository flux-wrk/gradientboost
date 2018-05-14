#include "BoostedClassifier.h"

namespace NGradientBoost {

    BoostedClassifier& BoostedClassifier::Fit(const Dataset& data, const Target& target) {
        auto begin_ts = std::chrono::high_resolution_clock::now();

        auto dataframe = DataFrame(data);
        trees_.clear();
        std::vector<float_t> current_approximation(dataframe.size(), 0.0f), next_approximation(dataframe.size(), 0);

        for (size_t iteration = 0; iteration < tree_count_; ++iteration) {
            DecisionTree weak_classifier(tree_depth_);
            weak_classifier.Fit(dataframe, target, current_approximation, next_approximation);

            for (size_t i = 0; i < dataframe.size(); ++i) {
                current_approximation[i] += learning_rate_ * next_approximation[i];
            }

            float_t loss = BoostedClassifier::MSE(target, current_approximation);
            std::cout << "Loss on iteration " << (iteration + 1) << " : " << loss << std::endl;

            trees_.push_back(weak_classifier);
        }
        auto end_ts = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_ts - begin_ts).count() / 1000.f;

        std::cout << std::endl;
        std::cout << "Fitting finished, time elapsed: " << duration << " seconds (" << duration / tree_count_
                  << " per tree)\n";
        return *this;
    }

    std::vector<float_t> BoostedClassifier::Predict(const std::vector<std::vector<float_t>>& data) const {
        auto dataframe = DataFrame(data);
        std::vector<float_t> predictions(dataframe.size());
        for (const DecisionTree& weak_clf : trees_) {
            std::vector<float_t> predictions_for_tree = weak_clf.Predict(data);
            for (size_t i = 0; i < predictions.size(); ++i) {
                predictions[i] += learning_rate_ * predictions_for_tree[i];
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

