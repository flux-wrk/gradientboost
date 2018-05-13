#include "Tree.h"

namespace NGradientBoost {

    DecisionTree::DecisionTree(size_t depth) : depth_(depth) {
        splitting_features_ = std::vector<size_t>();
        leaf_answers_ = std::vector<float_t>(1ul << depth_, 0.0);
    }

    std::vector<float_t>
    DecisionTree::Predict(const std::vector<std::vector<float_t>>& data) const {
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
    void DecisionTree::Save(std::ostream& stream) const {
        stream << depth_ << std::endl;
        stream << leaf_answers_.size() << std::endl;
        for (const auto& answer : leaf_answers_) {
            stream << answer << " ";
        }
        stream << std::endl << splitting_features_.size() << std::endl;

        for (auto feature : splitting_features_) {
            stream << feature << " ";
        }
        stream << std::endl;
    }

    DecisionTree::DecisionTree(std::istream& stream) {
        stream >> depth_;
        size_t leaf_answer_length;
        stream >> leaf_answer_length;
        leaf_answers_ = std::vector<float_t>(leaf_answer_length);
        for (size_t i = 0; i < leaf_answer_length; ++i) {
            stream >> leaf_answers_[i];
        }
        size_t splitting_feature_length;
        stream >> splitting_feature_length;
        splitting_features_ = std::vector<size_t>(splitting_feature_length);
        for (size_t i = 0; i < splitting_feature_length; ++i) {
            stream >> splitting_features_[i];
        }
    }

    void DecisionTree::Fit(const DataFrame& data, const Target& target) {

    }


}