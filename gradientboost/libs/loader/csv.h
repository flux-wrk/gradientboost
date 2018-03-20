#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cfloat>
#include <cmath>

namespace NGradientBoost {

std::vector<std::vector<float_t>> ReadCSV(std::istream& stream, char separator = ',', bool skip_header = true);
std::vector<std::vector<float_t>> ReadCSV(const std::string& file, char separator = ',', bool skip_header = true);

} // namespace NGradientBoost

