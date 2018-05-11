#include <memory>
#include "lib/loader/CSV.h"
#include "lib/preprocessing/Preprocessor.h"
#include "lib/boosting/BoostedClassifier.h"
#include "CLI/CLI.hpp"

using namespace NGradientBoost;


int main(int argc, char* argv[]) {
    std::unique_ptr<BoostedClassifier> classifier;

    CLI::App app {"Gradient boosting trainer."};
    CLI::App* fit = app.add_subcommand("fit", "Trains model on given dataset");
    std::string train_dataset_file, train_target_label = "label", model_file;
    size_t tree_depth = 4, trees_count = 16;
    float_t learning_rate = 1.0f;

    fit->add_option("--data", train_dataset_file, "Dataset file to train on")->required();
    fit->add_option("--target", train_target_label, "Target label")->required();
    fit->add_option("--trees", trees_count, "Number of trees in ensemble");
    fit->add_option("--depth", tree_depth, "Depth of each tree");
    fit->add_option("--model", model_file, "Name of saved model file")->required();

    fit->set_callback([&](){
        std::cout << "Called fit" << std::endl;
        classifier = std::make_unique<BoostedClassifier>(trees_count, tree_depth, learning_rate);
        // classifier->Fit();
        {
            std::ofstream stream(model_file);
            classifier->Save(stream);
        }
    });

    CLI::App* eval = app.add_subcommand("eval", "Evaluates model performance");
    std::string test_file, target_test_label = "label";
    eval->add_option("--data", test_file, "Model file for a classifier")->required();
    eval->add_option("--target", train_target_label, "Target label")->required();
    eval->set_callback([&](){
        std::cout << "Called eval" << std::endl;
        {
            std::ifstream stream(model_file);
            classifier = std::make_unique<BoostedClassifier>(stream);
        }
    });

    app.require_subcommand();
    CLI11_PARSE(app, argc, argv);

    return 0;
/*

    const auto train_frame = NGradientBoost::ReadCSV(params.train);
    std::cout << "Readed " << train_frame.size() << " row(s)" << std::endl;
    const auto test_frame = NGradientBoost::ReadCSV(params.test);
    std::cout << "Readed " << test_frame.size() << " row(s)" << std::endl;

    size_t dim = train_frame[0].size();

    std::vector<float_t> res;
    NGradientBoost::CategorialEncoder<std::string> label_encoder;
    NGradientBoost::ToFloat features_encoder;
    label_encoder.Fit(train_frame, dim - 1);

    const auto train_frame = NGradientBoost::ReadCSV(params.train);
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
    */
}
