#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <utility>
#include <iostream>
#include <cfloat>

namespace NGradientBoost {

    class DataFrame {
        std::vector<std::vector<float_t>> data_;
        std::vector<std::vector<bool>> binary_data_;
        std::vector<std::vector<float_t>> thresholds_;
        size_t num_features_{};
        const size_t BIN_COUNT = 16;

    public:
        DataFrame() {}

        explicit DataFrame(const std::vector<std::vector<float_t>>& data) {
            this->data_ = data;
            num_features_ = data[0].size();
            BinarizeData();
        }

        void BinarizeData() {
            binary_data_ = std::vector<std::vector<bool> >(data_.size(), std::vector<bool>(16 * num_features_));
            thresholds_ = std::vector<std::vector<float_t > >(num_features_, std::vector<float_t >(16));

            for (int j = 0; j < num_features_; ++j) {
                std::vector<double> feature_values(data_.size());
                for (int i = 0; i < data_.size(); ++i) {
                    feature_values[i] = data_[i][j];
                }

                std::sort(feature_values.begin(), feature_values.end());

                for (size_t i = 0; i < BIN_COUNT - 1; ++i) {
                    thresholds_[j][i] = static_cast<float_t>(feature_values[(i + 1) * data_.size() / BIN_COUNT]);
                }
                thresholds_[j][BIN_COUNT - 1] = static_cast<float_t>(DBL_MAX);

                for (int i = 0; i < data_.size(); ++i) {
                    for (int l = 0; l < BIN_COUNT; ++l) {
                        binary_data_[i][BIN_COUNT * j + l] = data_[i][j] < thresholds_[j][l];
                    }
                }
            }

        }

        size_t get_bin_count() const {
            return BIN_COUNT;
        }

        unsigned long get_size() const {
            return binary_data_.size();
        }

        std::vector<bool> operator[](int index) const {
            return binary_data_[index];
        }

        size_t features_count() const {
            return num_features_ * BIN_COUNT;
        }

    };

} // namespace NGradientBoost