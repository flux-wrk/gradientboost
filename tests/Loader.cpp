#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>
#include <string>

#include "lib/loader/CSV.h"

using namespace NGradientBoost;

TEST(Loader, Csv) {
    std::stringstream csv_stream;
    csv_stream
        << "name;col1;col2" << std::endl
        << "ok;1;2" << std::endl
        << "test;3;4" << std::endl;

    auto expected = std::vector<StringVector>{{"ok", "1", "2"}, {"test", "3", "4"}};
    auto result = ReadCSV(csv_stream, ';', true);

    EXPECT_THAT(result, ::testing::ContainerEq(expected));
}

