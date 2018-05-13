#include "CSV.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>

namespace NGradientBoost {

std::pair<StringMatrix, StringVector> ReadCSV(std::istream& stream, char separator, bool has_header) {
    StringMatrix result;
    StringVector header;

    std::string current_line;
    size_t last_vector_size = 0;
    while (std::getline(stream, current_line)) {
        std::stringstream lineStream(current_line);
        std::string cell;
        std::vector<std::string> current_vector;
        current_vector.reserve(last_vector_size);

        // TODO: handle escape codes
        while (std::getline(lineStream, cell, separator)) {
            current_vector.push_back(cell);
        }

        last_vector_size = current_vector.size();
        if (has_header) {
            header = current_vector;
            has_header = false;
        } else {
            result.emplace_back(current_vector);
        }
    }

    return {result, header};
}

std::pair<StringMatrix, StringVector> ReadCSV(const std::string& file, char separator, bool skip_header) {
    std::ifstream stream(file);
    if (!stream.good()) {
        throw std::ifstream::failure("IO error while reading csv (" + file + ").");
    }
    return ReadCSV(stream, separator, skip_header);
}

} // namespace NGradientBoost

