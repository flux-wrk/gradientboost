#pragma once

#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <utility>
#include <cfloat>
#include <limits>
#include "tbb/parallel_sort.h"

namespace NGradientBoost {

using Feature = float_t;
using Sample = std::vector<Feature>;
using Dataset = std::vector<Sample>;

using Label = float_t;
using Target = std::vector<Label>;

class DataFrame {
public:
    DataFrame() = default;

    explicit DataFrame(const Dataset& data) {
        this->data_ = data;
        num_features_ = data[0].size();
        DistributeToBins();
    }

    void DistributeToBins() {
        binary_data_ = std::vector<std::vector<bool>>(data_.size(), std::vector<bool>(BIN_COUNT * num_features_));
        thresholds_ = std::vector<std::vector<float_t>>(num_features_, std::vector<float_t >(BIN_COUNT));

        for (size_t j = 0; j < num_features_; ++j) {
            std::vector<Feature> feature_values(data_.size());
            for (size_t i = 0; i < data_.size(); ++i) {
                feature_values[i] = data_[i][j];
            }

            tbb::parallel_sort(std::begin(feature_values), std::end(feature_values));

            for (size_t i = 0; i < BIN_COUNT - 1; ++i) {
                thresholds_[j][i] = static_cast<float_t>(feature_values[(i + 1) * data_.size() / BIN_COUNT]);
            }
            thresholds_[j][BIN_COUNT - 1] = std::numeric_limits<float_t>::max();

            for (size_t i = 0; i < data_.size(); ++i) {
                for (size_t l = 0; l < BIN_COUNT; ++l) {
                    binary_data_[i][BIN_COUNT * j + l] = data_[i][j] < thresholds_[j][l];
                }
            }
        }

    }

    size_t get_bin_count() const {
        return BIN_COUNT;
    }

    size_t size() const {
        return binary_data_.size();
    }

    const std::vector<bool>& operator[](int index) const {
        return binary_data_[index];
    }

    size_t features_count() const {
        return num_features_ * BIN_COUNT;
    }

private:
    Dataset data_;
    std::vector<std::vector<bool>> binary_data_;
    std::vector<std::vector<float_t>> thresholds_;
    size_t num_features_{};
    const size_t BIN_COUNT = 16;
};

} // namespace NGradientBoost
