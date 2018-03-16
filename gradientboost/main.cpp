#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <iostream>

#define NOMINMAX

#include <tbb/parallel_for.h>
#include <tbb/parallel_do.h>
#include <tbb/mutex.h>
#include <tbb/task_scheduler_init.h>

int main(/*int argc, char** argv*/) {
    tbb::mutex mutex;
    std::vector<std::string> sample_strings = {"1", "2", "3"};
    tbb::parallel_do(sample_strings, [&](const std::string& str) {
        mutex.lock();
        std::cout << str << std::endl;
        mutex.unlock();
    });

    return 0;
}
