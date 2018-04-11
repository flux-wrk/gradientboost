#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cfloat>
#include <cmath>

namespace NGradientBoost {
    std::vector<std::vector<std::string>> ReadCSV(std::istream& stream, char separator = ',', bool skip_header = true);
    std::vector<std::vector<std::string>> ReadCSV(const std::string& file, char separator = ',', bool skip_header = true);
} // namespace NGradientBoost

