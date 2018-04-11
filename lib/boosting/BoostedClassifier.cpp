#include "BoostedClassifier.h"

namespace NGradientBoost {

BoostedClassifier::DecisionTree::DecisionTree(size_t depth) : depth_(depth) {
    splitting_features_ = std::vector<size_t>();
    leaf_answers_ = std::vector<float_t>(static_cast<unsigned long>(1 << (depth_ + 1)), 0.0);
}

std::vector<float_t>
BoostedClassifier::DecisionTree::Predict(const std::vector<std::vector<float_t>> &data) const {
    auto dataframe = DataFrame(data);
    std::vector<float_t> res(dataframe.size());
    for (size_t i = 0; i < dataframe.size(); ++i) {
        size_t mask = 0;
        for (size_t j = 0; j < depth_; ++j) {
            mask += (dataframe[i][splitting_features_[j]] << (depth_ - j));
        }
        res[i] = leaf_answers_[mask];
    }
    return res;
}

BoostedClassifier &
BoostedClassifier::Fit(const std::vector<std::vector<float_t>>& data, const std::vector<float_t> target) {
    auto dataframe = DataFrame(data);
    trees_.clear();
    std::vector<float_t> current_predictions(dataframe.size(), 0), temp_pred(dataframe.size(), 0);

    for (size_t iteration = 0; iteration < tree_count_; ++iteration) {
        DecisionTree weak_classifier(tree_depth_);
        std::vector<int> leaf_indices(dataframe.size(), 0);

        for (size_t depth = 0; depth < tree_depth_; ++depth) {
            std::vector<int> temp_leaf_ind(dataframe.size(), 0), best_leaf_ind(dataframe.size(), 0);
            std::vector<float_t> leaf_ans(1 << (depth + 1), 0), best_leaf_ans(1 << (depth + 1), 0);
            std::set<size_t> used_features;
            size_t best_feature = 0;
            auto best_mse = std::numeric_limits<float_t>::max();
            std::vector<float_t> best_leaf_sum;
            std::vector<int> best_leaf_count;

            for (size_t feature_index = 0; feature_index < dataframe.features_count(); ++feature_index) {
                std::vector<float_t> leaf_sum(static_cast<unsigned long>(1 << (depth + 1)), 0.0);
                std::vector<int> leaf_count(1 << (depth + 1), 0);
                float_t this_mse = 0.0;
                if (used_features.count(feature_index / dataframe.get_bin_count()) > 0) {
                    continue;
                }
                for (size_t i = 0; i < dataframe.size(); ++i) {
                    temp_leaf_ind[i] = leaf_indices[i] * 2 + dataframe[i][feature_index];
                    leaf_sum[temp_leaf_ind[i]] += target[i] - current_predictions[i];
                    ++leaf_count[temp_leaf_ind[i]];
                }

                bool is_good = true;
                for (size_t i = 0; i < leaf_ans.size(); ++i) {
                    if (leaf_count[i] == 0) {
                        is_good = false;
                        break;
                    }
                    leaf_ans[i] = leaf_sum[i] / leaf_count[i];
                    this_mse += leaf_ans[i] * (leaf_count[i] * leaf_ans[i] - 2 * leaf_sum[i]);
                }

                if (this_mse < best_mse && is_good) {
                    best_mse = this_mse;
                    best_feature = feature_index;
                    best_leaf_ans = leaf_ans;
                    best_leaf_ind = temp_leaf_ind;
                    best_leaf_sum = leaf_sum;
                    best_leaf_count = leaf_count;
                }
            }

            weak_classifier.splitting_features_.push_back(best_feature);
            weak_classifier.leaf_answers_ = best_leaf_ans;
            used_features.insert(best_feature / dataframe.get_bin_count());
            leaf_indices = best_leaf_ind;

            for (size_t i = 0; i < dataframe.size(); ++i) {
                temp_pred[i] = current_predictions[i] + best_leaf_ans[best_leaf_ind[i]];
            }
        }
        current_predictions = temp_pred;

        float_t MSE = 0.0;
        for (size_t i = 0; i < dataframe.size(); ++i) {
            MSE += (target[i] - temp_pred[i]) * (target[i] - temp_pred[i]);
        }
        MSE /= dataframe.size();
        std::cout << "MSE loss on iteration " << iteration << " : " << MSE << std::endl;

        trees_.push_back(weak_classifier);
    }
    return *this;
}

std::vector<float_t> BoostedClassifier::Predict(const std::vector<std::vector<float_t>> &data) const {
    auto dataframe = DataFrame(data);
    std::vector<float_t> predictions(dataframe.size());
    for (const DecisionTree &weak_clf : trees_) {
        std::vector<float_t> predictions_for_tree = weak_clf.Predict(data);
        for (size_t i = 0; i < predictions.size(); ++i) {
            predictions[i] += predictions_for_tree[i];
        }
    }
    return predictions;
}

} // namespace NGradientBoost

