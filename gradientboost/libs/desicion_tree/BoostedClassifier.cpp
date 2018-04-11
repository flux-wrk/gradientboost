#include "BoostedClassifier.h"

NGradientBoost::BoostedClassifier::DecisionTree::DecisionTree(size_t depth) : depth_(depth) {
    splitting_features_ = std::vector<size_t>();
    leaf_answers_ = std::vector<float_t>(static_cast<unsigned long>(1 << (depth_ + 1)), 0.0);
}

std::vector<float_t>
NGradientBoost::BoostedClassifier::DecisionTree::Predict(const std::vector<std::vector<float_t>> &data) const {
    auto ds = DataFrame(data);
    std::vector<float_t> res(ds.get_size());
    for (int i = 0; i < ds.get_size(); ++i) {
        size_t mask = 0;
        for (int j = 0; j < depth_; ++j) {
            mask += (ds[i][splitting_features_[j]] << (depth_ - j));
        }
        res[i] = leaf_answers_[mask];
    }
    return res;
}

NGradientBoost::BoostedClassifier &
NGradientBoost::BoostedClassifier::Fit(const std::vector<std::vector<float_t>> &data, const std::vector<float_t> target) {
    auto dataset = DataFrame(data);
    trees_.clear();
    std::vector<float_t> cur_pred(dataset.get_size(), 0), temp_pred(dataset.get_size(), 0);

    for (size_t iteration = 0; iteration < tree_count_; ++iteration) {
        DecisionTree weak_classifier(tree_depth_);
        std::vector<int> leaf_ind(dataset.get_size(), 0);

        for (size_t depth = 0; depth < tree_depth_; ++depth) {
            std::vector<int> temp_leaf_ind(dataset.get_size(), 0), best_leaf_ind(dataset.get_size(), 0);
            std::vector<float_t> leaf_ans(1 << (depth + 1), 0), best_leaf_ans(1 << (depth + 1), 0);
            std::set<size_t> used_features;
            size_t best_feature = 0;
            auto best_mse = std::numeric_limits<float_t>::max();
            std::vector<float_t> best_leaf_sum;
            std::vector<int> best_leaf_count;

            for (size_t feature_index = 0; feature_index < dataset.features_count(); ++feature_index) {
                std::vector<float_t> leaf_sum(static_cast<unsigned long>(1 << (depth + 1)), 0.0);
                std::vector<int> leaf_count(1 << (depth + 1), 0);
                float_t this_mse = 0.0, this_true_mse = 0.0;
                if (used_features.count(feature_index / dataset.get_bin_count()) > 0) {
                    continue;
                }
                for (int i = 0; i < dataset.get_size(); ++i) {
                    temp_leaf_ind[i] = leaf_ind[i] * 2 + dataset[i][feature_index];
                    leaf_sum[temp_leaf_ind[i]] += target[i] - cur_pred[i];
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
            used_features.insert(best_feature / dataset.get_bin_count());
            leaf_ind = best_leaf_ind;

            for (int i = 0; i < dataset.get_size(); ++i) {
                temp_pred[i] = cur_pred[i] + best_leaf_ans[best_leaf_ind[i]];
            }
        }
        cur_pred = temp_pred;

        float_t MSE = 0.0;
        for (int i = 0; i < dataset.get_size(); ++i) {
            MSE += (target[i] - temp_pred[i]) * (target[i] - temp_pred[i]);
        }
        MSE /= dataset.get_size();
        std::cout << "MSE loss on iteration " << iteration << " : " << MSE << std::endl;

        trees_.push_back(weak_classifier);
    }
    return *this;
}

std::vector<float_t> NGradientBoost::BoostedClassifier::Predict(const std::vector<std::vector<float_t>> &data) const {
    auto dataset = DataFrame(data);
    std::vector<float_t> predictions(dataset.get_size());
    for (const DecisionTree &weak_clf : trees_) {
        std::vector<float_t> predictions_for_tree = weak_clf.Predict(data);
        for (int i = 0; i < predictions.size(); ++i) {
            predictions[i] += predictions_for_tree[i];
        }
    }
    return predictions;
}
