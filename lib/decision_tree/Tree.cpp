#include "Tree.h"

namespace NGradientBoost {

    DecisionTree::DecisionTree(size_t depth) : depth_(depth) {
        splitting_features_ = std::vector<size_t>();
        leaf_answers_ = std::vector<float_t>(1ul << depth_, 0.0);
    }

    void DecisionTree::Save(std::ostream& stream) const {
        stream << depth_ << " ";
        for (auto feature : splitting_features_) {
            stream << feature << " ";
        }
        for (const auto& answer : leaf_answers_) {
            stream << answer << " ";
        }
        stream << std::endl;
    }

    DecisionTree::DecisionTree(std::istream& stream) {
        stream >> depth_;

        splitting_features_.resize(depth_);
        for (size_t i = 0; i < depth_; ++i) {
            stream >> splitting_features_[i];
        }

        leaf_answers_.resize(1ul << depth_);
        for (size_t i = 0; i < (1ul << depth_); ++i) {
            stream >> leaf_answers_[i];
        }
    }

    std::vector<float_t> DecisionTree::Predict(const std::vector<std::vector<float_t>>& data) const {
        auto dataframe = DataFrame(data);
        std::vector<float_t> res(dataframe.size());
        for (size_t i = 0; i < dataframe.size(); ++i) {
            size_t mask = 0;
            for (size_t j = 0; j < depth_; ++j) {
                mask += (dataframe[i][splitting_features_[j]] << (depth_ - j - 1));
            }
            res[i] = leaf_answers_[mask];
        }
        return res;
    }

    void DecisionTree::Fit(const DataFrame& dataframe,
                           const Target& target,
                           Target& current_predictions,
                           Target& temp_predictions) {
        std::vector<int> leaf_indices(dataframe.size(), 0);
        tbb::mutex locker;

        for (size_t depth = 1; depth <= depth_; ++depth) {
            size_t layer_width = 1ul << depth;
            std::vector<int> chosen_leaf_ind(dataframe.size(), 0);
            std::vector<float_t> chosen_leaf_ans(layer_width, 0);
            std::set<size_t> used_features;
            size_t chosen_feature = 0;
            float_t chosen_mse = std::numeric_limits<float_t>::max();
            std::vector<float_t> chosen_leaf_sum;
            std::vector<int> chosen_leaf_count;

            tbb::parallel_for(size_t(0), dataframe.features_count(), size_t(1), [&](size_t feature_index) {
                std::vector<int> temp_leaf_ind(dataframe.size(), 0), leaf_count(layer_width, 0);
                std::vector<float_t> leaf_ans(layer_width, 0.0f), leaf_sum(layer_width, 0.0f);

                if (used_features.count(feature_index / dataframe.get_bin_count()) > 0) {
                    return;
                }

                for (size_t i = 0; i < dataframe.size(); ++i) {
                    temp_leaf_ind[i] = leaf_indices[i] * 2 + dataframe[i][feature_index];
                    leaf_sum[temp_leaf_ind[i]] += target[i] - current_predictions[i];
                    ++leaf_count[temp_leaf_ind[i]];
                }

                float_t current_mse = 0.0;
                for (size_t i = 0; i < leaf_ans.size(); ++i) {
                    leaf_ans[i] = (leaf_count[i] == 0) ? 0 : leaf_sum[i] / leaf_count[i];
                    current_mse += leaf_ans[i] * (leaf_count[i] * leaf_ans[i] - 2 * leaf_sum[i]);
                }

                {
                    tbb::mutex::scoped_lock lock(locker);
                    if (current_mse < chosen_mse) {
                        chosen_mse = current_mse;
                        chosen_feature = feature_index;
                        chosen_leaf_ans.swap(leaf_ans);
                        chosen_leaf_ind.swap(temp_leaf_ind);
                        chosen_leaf_sum.swap(leaf_sum);
                        chosen_leaf_count.swap(leaf_count);
                    }
                }
            });

            splitting_features_.push_back(chosen_feature);
            leaf_answers_ = chosen_leaf_ans;
            used_features.insert(chosen_feature / dataframe.get_bin_count());
            leaf_indices = chosen_leaf_ind;

            for (size_t i = 0; i < dataframe.size(); ++i) {
                temp_predictions[i] = current_predictions[i] + chosen_leaf_ans[chosen_leaf_ind[i]];
            }
        }
    }

} //namespace NGradientBoost
