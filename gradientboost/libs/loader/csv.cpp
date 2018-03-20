#include "csv.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace NGradientBoost {

std::vector<std::vector<std::string>> ReadCSV(std::istream& stream, char separator, bool skip_header) {
    std::vector<std::vector<std::string>> result = {};

    std::string current_line;
    while (std::getline(stream, current_line)) {
        if (skip_header) {
            skip_header = false;
            continue;
        }

        std::stringstream lineStream(current_line);
        std::string cell;
        std::vector<std::string> current_vector;

        // TODO: handle escape codes
        while (std::getline(lineStream, cell, separator)) {
            current_vector.push_back(cell);
        }
        result.push_back(current_vector);
    }

    return result;
}

std::vector<std::vector<std::string>> ReadCSV(const std::string& file, char separator, bool skip_header) {
    std::ifstream stream(file);
    if (!stream.good()) {
        throw std::ifstream::failure("IO error while reading csv (" + file + ").");
    }
    const auto result = ReadCSV(stream, separator, skip_header);
    return result;
}

} // namespace NGradientBoost

