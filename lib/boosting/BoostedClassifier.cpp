#include "BoostedClassifier.h"

namespace NGradientBoost {

BoostedClassifier::DecisionTree::DecisionTree(size_t depth) : depth_(depth) {
    splitting_features_ = std::vector<size_t>();
    leaf_answers_ = std::vector<float_t>(1ul << depth_, 0.0);
}

std::vector<float_t>
BoostedClassifier::DecisionTree::Predict(const std::vector<std::vector<float_t>> &data) const {
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

BoostedClassifier& BoostedClassifier::Fit(const Dataset& data, const Target& target) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    auto dataframe = DataFrame(data);
    trees_.clear();
    std::vector<float_t> current_predictions(dataframe.size(), 0), temp_pred(dataframe.size(), 0);

    for (size_t iteration = 0; iteration < tree_count_; ++iteration) {
        DecisionTree weak_classifier(tree_depth_);
        std::vector<int> leaf_indices(dataframe.size(), 0);
        tbb::mutex locker;

        for (size_t depth = 1; depth <= tree_depth_; ++depth) {
            std::vector<int> best_leaf_ind(dataframe.size(), 0);
            std::vector<float_t> best_leaf_ans(1ul << depth, 0);
            std::set<size_t> used_features;
            size_t best_feature = 0;
            auto best_mse = std::numeric_limits<float_t>::max();
            std::vector<float_t> best_leaf_sum;
            std::vector<int> best_leaf_count;

            tbb::parallel_for(size_t(0), dataframe.features_count(), size_t(1), [&](size_t feature_index) {
                std::vector<int> temp_leaf_ind(dataframe.size(), 0);
                std::vector<float_t> leaf_ans(1ul << depth, 0);
                std::vector<float_t> leaf_sum(1ul << depth, 0.0);
                std::vector<int> leaf_count(1ul << depth, 0);

                float_t this_mse = 0.0;
                if (used_features.count(feature_index / dataframe.get_bin_count()) > 0) {
                    return;
                }
                for (size_t i = 0; i < dataframe.size(); ++i) {
                    temp_leaf_ind[i] = leaf_indices[i] * 2 + dataframe[i][feature_index];
                    leaf_sum[temp_leaf_ind[i]] += target[i] - current_predictions[i];
                    ++leaf_count[temp_leaf_ind[i]];
                }

                for (size_t i = 0; i < leaf_ans.size(); ++i) {
                    if (leaf_count[i] == 0) {
                        leaf_ans[i] = 0;
                    } else {
                        leaf_ans[i] = leaf_sum[i] / leaf_count[i];
                    }
                    this_mse += leaf_ans[i] * (leaf_count[i] * leaf_ans[i] - 2 * leaf_sum[i]);
                }

                {
                    tbb::mutex::scoped_lock lock(locker);
                    if (this_mse < best_mse) {
                        best_mse = this_mse;
                        best_feature = feature_index;
                        best_leaf_ans = leaf_ans;
                        best_leaf_ind = temp_leaf_ind;
                        best_leaf_sum = leaf_sum;
                        best_leaf_count = leaf_count;
                    }
                }
            });

            weak_classifier.splitting_features_.push_back(best_feature);
            weak_classifier.leaf_answers_ = best_leaf_ans;
            used_features.insert(best_feature / dataframe.get_bin_count());
            leaf_indices = best_leaf_ind;

            for (size_t i = 0; i < dataframe.size(); ++i) {
                temp_pred[i] = current_predictions[i] + best_leaf_ans[best_leaf_ind[i]];
            }
        }
        current_predictions = temp_pred;

        std::cout << "MSE loss on iteration " << iteration << " : " << BoostedClassifier::MSE(target, temp_pred) << std::endl;

        trees_.push_back(weak_classifier);
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "\nFitting complete, time elapsed: " << duration << " seconds\n";

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
    bool BoostedClassifier::Save(std::ostream& stream) const {
        stream << tree_depth_ << " " << tree_count_ << " " << learning_rate_ << std::endl;
        for (auto tree : trees_) {
            stream << tree.depth_ << std::endl;
            stream << tree.leaf_answers_.size() << std::endl;
            for (auto answer : tree.leaf_answers_) {
                stream << answer << " ";
            }
            stream << std::endl << tree.splitting_features_.size() << std::endl;

            for (auto feature : tree.splitting_features_) {
                stream << feature << " ";
            }
            stream << std::endl;
        }
        return true;
    }

    BoostedClassifier::BoostedClassifier(std::istream& stream) {
        stream >> tree_depth_ >> tree_count_ >> learning_rate_;
        for (size_t idx = 0; idx < tree_count_; ++idx) {
            size_t depth;
            stream >> depth;
            DecisionTree new_classifier(depth);
            size_t leaf_answer_length;
            stream >> leaf_answer_length;
            new_classifier.leaf_answers_ = std::vector<float_t>(leaf_answer_length);
            for (size_t i = 0; i < leaf_answer_length; ++i) {
                stream >> new_classifier.leaf_answers_[i];
            }
            size_t splitting_feature_length;
            stream >> splitting_feature_length;
            new_classifier.splitting_features_ = std::vector<size_t>(splitting_feature_length);
            for (size_t i = 0; i < splitting_feature_length; ++i) {
                stream >> new_classifier.splitting_features_[i];
            }
            trees_.push_back(new_classifier);
        }
    }

    float_t BoostedClassifier::Eval(const Dataset& data, const Target& target) {
        return BoostedClassifier::MSE(Predict(data), target);
    }

    inline float_t sqr(float_t x) { return  x * x; }
    float_t BoostedClassifier::MSE(const Target& predicted, const Target& actual) {
        for (int i = 0; i < 10; ++i) {
            //std::cout << predicted[i] << " " << actual[i] << std::endl;
        }

        assert(predicted.size() == actual.size());
        float_t MSE = 0.0;
        for (size_t idx = 0; idx < predicted.size(); ++idx) {
            MSE += sqr(predicted[idx] - actual[idx]);
        }
        return MSE / predicted.size();
    }

} // namespace NGradientBoost

