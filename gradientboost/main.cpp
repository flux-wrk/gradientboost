#include "lib/loader/CSV.h"
#include "lib/utils/ParseCmd.h"
#include "lib/preprocessing/Preprocessor.h"
#include "lib/boosting/BoostedClassifier.h"

int main(int argc, char* argv[]) {
    const auto params = NGradientBoost::ParseCmd(argc, argv);

    const auto train_frame = NGradientBoost::ReadCSV(params.train);
    std::cout << "Readed " << train_frame.size() << " row(s)" << std::endl;
    const auto test_frame = NGradientBoost::ReadCSV(params.test);
    std::cout << "Readed " << test_frame.size() << " row(s)" << std::endl;

    size_t dim = train_frame[0].size();

    std::vector<float_t> res;
    NGradientBoost::CategorialEncoder<std::string> label_encoder;
    NGradientBoost::ToFloat features_encoder;
    label_encoder.Fit(train_frame, dim - 1);

    std::vector<float_t> train_target;
    std::vector<std::vector<float_t>> train_features;
    for (const auto& row : train_frame) {
        std::vector<float_t> feature_vec;
        for (size_t i = 0; i < dim - 1; ++i) {
            features_encoder.Transform(row[i], feature_vec);
        }
        train_features.emplace_back(feature_vec);
        label_encoder.Transform(row[dim - 1], train_target);
    }

    std::vector<std::vector<float_t>> test_features;
    size_t dim_test = test_frame[0].size();
    for (const auto& row : test_frame) {
        std::vector<float_t> feature_vec;
        for (size_t i = 0; i < dim_test; ++i) {
            features_encoder.Transform(row[i], feature_vec);
        }
        test_features.emplace_back(feature_vec);
    }

    NGradientBoost::BoostedClassifier classifier(params.learning_rate, params.tree_depth, params.tree_count);
    auto predictions = classifier.Fit(train_features, train_target).Predict(test_features);

    return 0;
}
