#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cfloat>
#include <cmath>

namespace NGradientBoost {

using StringVector = std::vector<std::string>;
using StringMatrix = std::vector<StringVector>;

    std::pair<StringMatrix, StringVector> ReadCSV(std::istream& stream, char separator = ',', bool has_header = true);
    std::pair<StringMatrix, StringVector> ReadCSV(const std::string& file, char separator = ',', bool skip_header = true);

} // namespace NGradientBoost

