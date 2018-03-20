#include "libs/loader/csv.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_do.h>
#include <tbb/mutex.h>
#include <tbb/task_scheduler_init.h>

int main() {
    tbb::mutex mutex;
    std::vector<std::string> sample_strings = {"1", "2", "3"};
    tbb::parallel_do(sample_strings, [&](const std::string& str) {
        mutex.lock();
        std::cout << str << std::endl;
        mutex.unlock();
    });

    const auto dataframe = NGradientBoost::ReadCSV("data/training.csv");
    std::cout << "Readed " << dataframe.size() << " row(s)" << std::endl;

    return 0;
}
