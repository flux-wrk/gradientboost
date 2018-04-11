#pragma once

#include <string>
#include <cmath>

namespace NGradientBoost {

struct CmdParams {
    std::string train;
    std::string test;
    float_t learning_rate;
    size_t tree_depth;
    size_t tree_count;
};

CmdParams ParseCmd(int argc, char *argv[]);

} // namespace NGradientBoost

