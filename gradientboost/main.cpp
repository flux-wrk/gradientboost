#include <vector>
#include <string>
#include <random>

#include <fstream>
#include <sstream>

#include <iostream>

#include <tbb/parallel_for.h>
#include <tbb/parallel_do.h>
#include <tbb/mutex.h>
#include <tbb/task_scheduler_init.h>


std::vector<std::vector<float_t>> ReadCSV(std::istream& stream, char separator = ',', bool skip_header = true) {
    std::vector<std::vector<float_t>> result = {};

    std::string current_line;
    while(std::getline(stream, current_line)) {
        if (skip_header) {
            skip_header = false;
            continue;
        }
        std::stringstream          lineStream(current_line);
        std::string                cell;
        std::vector<float_t> current_vector;
        std::cout << current_line << std::endl;

        while(std::getline(lineStream, cell, separator)) {
            try {
                current_vector.push_back(std::stof(cell));
            } catch (const std::exception&) {
                std::cout << cell << std::endl;
            }
        }
        result.push_back(current_vector);
    }

    return result;
}

std::vector<std::vector<float_t>> ReadCSV(const std::string& file, char separator = ',', bool skip_header = true) {
    std::ifstream stream(file);
    if (!stream.good()) {
        throw std::ifstream::failure("IO error while reading csv (" + file + ").");
    }
    auto result = ReadCSV(stream, separator, skip_header);
    stream.close();

    return result;
}


int main(int argc, char** argv) {
    tbb::mutex mutex;
    std::vector<std::string> sample_strings = {"1", "2", "3"};
    tbb::parallel_do(sample_strings, [&](const std::string& str) {
        mutex.lock();
        std::cout << str << std::endl;
        mutex.unlock();
    });

    auto dataframe = ReadCSV("data/training.csv");

    std::cout << dataframe.size() << std::endl;


    return 0;
}
