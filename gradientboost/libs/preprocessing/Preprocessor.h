#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <memory>

#include <set>
#include <map>


namespace NGradientBoost {
    template <typename TInput, typename TOutput>
    class Transformer {
     public:
        virtual Transformer<TInput, TOutput>& fit(const std::vector<std::vector<TInput>>& dataframe, size_t column) = 0;
        virtual void transform(const TInput& data, std::vector<TOutput>& output) = 0;
    };

    template <typename TInput>
    class OneHotEncoder : public Transformer<TInput, float_t> {
     public:
        OneHotEncoder<TInput>& fit(const std::vector<std::vector<TInput>>& dataframe, size_t column) override  {
            for (size_t row = 0; row < dataframe.size(); ++row) {
                all_objects_.insert(dataframe[row][column]);
            }
            return *this;
        }

        void transform(const TInput& data, std::vector<float_t>& output) override  {
            for(const auto& obj: all_objects_) {
                output.push_back(data == obj ? 1.0f : 0.0f);
            }
        }
     private:
        std::set<TInput> all_objects_;
    };

    template <typename TInput>
    class CategorialEncoder : public Transformer<TInput, float_t> {
    public:
        CategorialEncoder<TInput>& fit(const std::vector<std::vector<TInput>>& dataframe, size_t column) override  {
            for (size_t row = 0; row < dataframe.size(); ++row) {
                all_objects_.insert(dataframe[row][column]);
            }

            mapping_.clear();
            float_t category = 0.0f;
            for (const auto& value : all_objects_) {
                mapping_[value] = category;
                category += 1.0f;
            }

            return *this;
        }

        void transform(const TInput& data, std::vector<float_t>& output) override  {
            //TODO: handle absent elements
            output.push_back(mapping_[data]);
        }
    private:
        std::set<TInput> all_objects_;
        std::map<TInput, float_t> mapping_;
    };


    class ToFloat : public Transformer<std::string, float_t> {
     public:
        ToFloat& fit(const std::vector<std::vector<std::string>>& dataframe, size_t column) override {
            return *this;
        }

        void transform(const std::string& data, std::vector<float_t>& output) override {
            output.push_back(std::stof(data));
        }
    };

    template <typename TInput, typename TOutput>
    std::vector<std::vector<TOutput>> applyTransforms(const std::vector<std::vector<TInput>>& data,
                                                      const std::vector<Transformer<TInput, TOutput>>& transformers) {
        std::vector<std::vector<TOutput>> result;
        return result;
    }

} // namespace NGradientBoost

