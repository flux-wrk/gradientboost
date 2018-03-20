#include "csv.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace NGradientBoost {

std::vector<std::vector<float_t>> ReadCSV(std::istream& stream, char separator, bool skip_header) {
    std::vector<std::vector<float_t>> result = {};

    std::string current_line;
    while (std::getline(stream, current_line)) {
        if (skip_header) {
            skip_header = false;
            continue;
        }
        std::cout << current_line << std::endl;

        std::stringstream lineStream(current_line);
        std::string cell;
        std::vector<float_t> current_vector;

        while (std::getline(lineStream, cell, separator)) {
            try {
                current_vector.push_back(std::stof(cell));
            } catch (const std::exception&) {
                // TODO: handle exception
                std::cout << cell << std::endl;
            }
        }
        result.push_back(current_vector);
    }

    return result;
}

std::vector<std::vector<float_t>> ReadCSV(const std::string& file, char separator, bool skip_header) {
    std::ifstream stream(file);
    if (!stream.good()) {
        throw std::ifstream::failure("IO error while reading csv (" + file + ").");
    }
    const auto result = ReadCSV(stream, separator, skip_header);
    return result;
}

} // namespace NGradientBoost

