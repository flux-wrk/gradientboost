#include <memory>
#include <algorithm>
#include "lib/loader/CSV.h"
#include "lib/preprocessing/Preprocessor.h"
#include "lib/boosting/BoostedClassifier.h"
#include "CLI/CLI.hpp"
#include "tbb/task_scheduler_init.h"

using namespace NGradientBoost;

std::pair<Dataset, Target> LoadDataset(const std::string& file_name, const std::string& target_column) {
    StringMatrix string_frame;
    StringVector csv_header;
    std::tie(string_frame, csv_header) = NGradientBoost::ReadCSV(file_name);

    auto header_position = std::find(std::cbegin(csv_header), std::cend(csv_header), target_column);
    size_t target_index = (header_position == std::cend(csv_header)) ? 0 : header_position - std::cbegin(csv_header);

    Target train_target;
    Dataset train_features;

    NGradientBoost::CategorialEncoder<std::string> label_encoder;
    NGradientBoost::ToFloat features_encoder;

    label_encoder.Fit(string_frame, target_index);

    for (const auto& row : string_frame) {
        std::vector<float_t> feature_vec;
        for (size_t i = 0; i < row.size(); ++i) {
            if (i != target_index) {
                features_encoder.Transform(row[i], feature_vec);
            }
        }

        train_features.emplace_back(feature_vec);
        label_encoder.Transform(row[target_index], train_target);
    }

    return {train_features, train_target};
};

int main(int argc, char* argv[]) {
    std::unique_ptr<BoostedClassifier> classifier;

    CLI::App app{"Gradient boosting trainer."};
    int num_threads = -1;
    app.add_option("--nthreads", num_threads, "Number of threads to use");

    CLI::App* fit = app.add_subcommand("fit", "Trains model on given dataset");
    std::string train_dataset_file, train_target_label = "Label", model_file;
    size_t tree_depth = 4, trees_count = 16;
    float_t learning_rate = 1.0f;

    fit->add_option("--data", train_dataset_file, "Dataset file to train on")->required()->check(CLI::ExistingFile);
    fit->add_option("--target", train_target_label, "Target label")->required();
    fit->add_option("--trees", trees_count, "Number of trees in ensemble");
    fit->add_option("--depth", tree_depth, "Depth of each tree");
    fit->add_option("--model", model_file, "Name of saved model file")->required();

    fit->set_callback([&]() {
        tbb::task_scheduler_init scheduler(num_threads);
        std::cout << "Called fit on " << train_dataset_file << ", loading data: ";

        Dataset train_features;
        Target train_target;
        std::tie(train_features, train_target) = LoadDataset(train_dataset_file, train_target_label);
        std::cout << train_target.size() << " rows" << std::endl;

        classifier = std::make_unique<BoostedClassifier>(trees_count, tree_depth, learning_rate);

        std::cout << "Fitting GBM with " << trees_count << " trees of depth " << tree_depth << ":" << std::endl;

        classifier->Fit(train_features, train_target);
        {
            std::ofstream stream(model_file);
            classifier->Save(stream);
        }
    });

    CLI::App* eval = app.add_subcommand("eval", "Evaluates model performance");
    std::string test_dataset_file, target_test_label = "label";
    eval->add_option("--data", test_dataset_file, "Model file for a classifier")->required()->check(CLI::ExistingFile);
    eval->add_option("--target", target_test_label, "Target label")->required();
    eval->add_option("--model", model_file, "Name of model file to test")->check(CLI::ExistingFile);

    eval->set_callback([&]() {
        tbb::task_scheduler_init scheduler(num_threads);
        std::cout << "Called eval" << std::endl;

        Dataset test_features;
        Target test_target;
        std::tie(test_features, test_target) = LoadDataset(test_dataset_file, target_test_label);

        if (!classifier) { // classifier is not fitted with "fit" subcommand
            std::ifstream stream(model_file);
            classifier = std::make_unique<BoostedClassifier>(stream);
        }

        auto mse = classifier->Eval(test_features, test_target);
        std::cout << "MSE: " << mse << std::endl;
    });

    app.require_subcommand();

    std::cout.precision(5);
    CLI11_PARSE(app, argc, argv);

    return 0;
}

/*
Loss on iteration 1 : 0.011618
Loss on iteration 2 : 0.010934
Loss on iteration 3 : 0.010557
Loss on iteration 4 : 0.0098072
Loss on iteration 5 : 0.0096362
*/
