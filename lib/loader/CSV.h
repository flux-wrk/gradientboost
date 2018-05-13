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

    std::pair<StringMatrix, StringVector> ReadCSV(const std::string& file,
                                                  char separator = ',',
                                                  bool skip_header = true);

    void WriteCSV(std::ostream& stream, const std::vector<float_t>& values, std::string header="");

} // namespace NGradientBoost

