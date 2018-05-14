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
            DistributeToSlots();
        }

        void DistributeToSlots() {
            binary_data_ = std::vector<std::vector<bool>>(data_.size(), std::vector<bool>(SLOT_COUNT * num_features_));
            thresholds_ = std::vector<std::vector<float_t>>(num_features_, std::vector<float_t >(SLOT_COUNT));

            for (size_t feature_idx = 0; feature_idx < num_features_; ++feature_idx) {
                std::vector<Feature> feature_values(data_.size());
                for (size_t sample_idx = 0; sample_idx < data_.size(); ++sample_idx) {
                    feature_values[sample_idx] = data_[sample_idx][feature_idx];
                }

                tbb::parallel_sort(std::begin(feature_values), std::end(feature_values));

                for (size_t bin_idx = 0; bin_idx < SLOT_COUNT - 1; ++bin_idx) {
                    thresholds_[feature_idx][bin_idx] = feature_values[(bin_idx + 1) * data_.size() / SLOT_COUNT];
                }

                thresholds_[feature_idx][SLOT_COUNT - 1] = std::numeric_limits<float_t>::max();

                for (size_t sample_idx = 0; sample_idx < data_.size(); ++sample_idx) {
                    for (size_t bin_idx = 0; bin_idx < SLOT_COUNT; ++bin_idx) {
                        binary_data_[sample_idx][SLOT_COUNT * feature_idx + bin_idx] =
                            data_[sample_idx][feature_idx] < thresholds_[feature_idx][bin_idx];
                    }
                }
            }
        }

        size_t slot_count() const {
            return SLOT_COUNT;
        }

        size_t size() const {
            return binary_data_.size();
        }

        const std::vector<bool>& operator[](int index) const {
            return binary_data_[index];
        }

        size_t features_count() const {
            return num_features_ * SLOT_COUNT;
        }

    private:
        Dataset data_;
        std::vector<std::vector<bool>> binary_data_;
        std::vector<std::vector<float_t>> thresholds_;
        size_t num_features_{};
        const size_t SLOT_COUNT = 16;
    };

} // namespace NGradientBoost
