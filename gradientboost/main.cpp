#include "CSV.h"
#include "Preprocessor.h"
#include "libs/desicion_tree/BoostedClassifier.h"
#include "../3d-party/cxxopts/include/cxxopts.hpp"

int main(int argc, char* argv[]) {
    cxxopts::Options options(argv[0], "Trains and applies gradient boosting classifier");
    options.add_options()
            ("train", "CSV file with train data set", cxxopts::value<std::string>())
            ("test", "CSV file with test data set", cxxopts::value<std::string>())
            ("n, num-trees", "Number of trees in boosted classifier", cxxopts::value<size_t >()->default_value("16"))
            ("d, tree-depth", "Depth of each tree", cxxopts::value<size_t>()->default_value("4"))
            ("l, learning-rate", "Learning rate", cxxopts::value<float_t >()->default_value("0.1"));
    options.parse(argc, argv);

    std::string train_file = options["train"].as<std::string>();
    std::string test_file = options["test"].as<std::string>();
    float_t learning_rate = options["learning-rate"].as<float_t>();
    size_t tree_depth = options["tree-depth"].as<size_t>();
    size_t tree_count = options["num-trees"].as<size_t>();

    const auto train_frame = NGradientBoost::ReadCSV(train_file);
    std::cout << "Readed " << train_frame.size() << " row(s)" << std::endl;
    const auto test_frame = NGradientBoost::ReadCSV(test_file);
    std::cout << "Readed " << test_frame.size() << " row(s)" << std::endl;

    size_t dim = train_frame[0].size();

    std::vector<float_t> res;
    NGradientBoost::CategorialEncoder<std::string> label_encoder;
    NGradientBoost::ToFloat features_encoder;
    label_encoder.fit(train_frame, dim - 1);

    std::vector<float_t> train_target;
    std::vector<std::vector<float_t>> train_features;
    int cnt = 0;
    for (const auto& row : train_frame) {
        std::vector<float_t> feature_vec;
        for (size_t i = 0; i < dim - 1; ++i) {
            features_encoder.transform(row[i], feature_vec);
        }
        train_features.emplace_back(feature_vec);
        label_encoder.transform(row[dim - 1], train_target);
        //if (++cnt == 20000) break;
    }

    std::vector<std::vector<float_t>> test_features;
    size_t dim_test = test_frame[0].size();
    for (const auto& row : test_frame) {
        std::vector<float_t> feature_vec;
        for (size_t i = 0; i < dim_test; ++i) {
            features_encoder.transform(row[i], feature_vec);
        }
        test_features.emplace_back(feature_vec);
    }

    NGradientBoost::BoostedClassifier classifier(learning_rate, tree_depth, tree_count);
    auto predictions = classifier.Fit(train_features, train_target).Predict(test_features);

    return 0;
}
