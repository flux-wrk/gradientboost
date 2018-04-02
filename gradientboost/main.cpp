#include "libs/loader/csv.h"
#include "libs/preprocessing/preprocessor.h"

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
    size_t dim = dataframe[0].size() - 1;
    std::cout << "Readed " << dataframe.size() << " row(s)" << std::endl;


    std::vector<float_t> res;
    NGradientBoost::OneHotEncoder<std::string> encoder;
    encoder.fit(dataframe, dim).fit(dataframe, dim);

    for(const auto& vec: dataframe) {
        std::cout << vec[dim] << " : ";
        std::cout << std::endl;
    }


    return 0;
}
