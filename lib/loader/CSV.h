#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cfloat>
#include <cmath>

namespace NGradientBoost {

using StringVector= std::vector<std::string>;

std::vector<StringVector> ReadCSV(std::istream& stream, char separator = ',', bool skip_header = true);
std::vector<StringVector> ReadCSV(const std::string& file, char separator = ',', bool skip_header = true);

} // namespace NGradientBoost

