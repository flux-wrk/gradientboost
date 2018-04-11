#include "ParseCmd.h"

#include "3d-party/cxxopts/include/cxxopts.hpp"

#include <cfloat>
#include <cmath>

namespace NGradientBoost {

CmdParams ParseCmd(int argc, char *argv[]) {
    cxxopts::Options options(argv[0], "Trains and applies gradient boosting classifier");
    options.add_options()
        ("train", "CSV file with train data set", cxxopts::value<std::string>())
        ("test", "CSV file with test data set", cxxopts::value<std::string>())
        ("n, num-trees", "Number of trees in boosted classifier", cxxopts::value<size_t >()->default_value("16"))
        ("d, tree-depth", "Depth of each tree", cxxopts::value<size_t>()->default_value("4"))
        ("l, learning-rate", "Learning rate", cxxopts::value<float_t >()->default_value("0.1"));
    options.parse(argc, argv);
    return CmdParams{
        options["train"].as<std::string>(),
        options["test"].as<std::string>(),
        options["learning-rate"].as<float_t>(),
        options["tree-depth"].as<size_t>(),
        options["num-trees"].as<size_t>(),
    };
}

} // namespace NGradientBoost

