#include "catch.hpp"
#include "../src/inputdatafilereader.h"
#include <filesystem>

using namespace std;
using namespace neuralnetworkfirstprinciples;

TEST_CASE("Read Kaggle formatted data and convert into vector of ColVectors", "[readkaggledata]") {

    string path = "E:/Code/kaggle/digits/test/";

    SECTION("Read as vector of vectors") 
    {
        string filename = "train_test.csv";
        vector<shared_ptr<ColVector>> labelAsVectors;
        vector<shared_ptr<ColVector>> digitisedNumbers;

        read_digit_data(path + filename, labelAsVectors, digitisedNumbers);

        REQUIRE(labelAsVectors.size() == digitisedNumbers.size());
        REQUIRE(labelAsVectors.size() == 301);
        REQUIRE(labelAsVectors[0]->rows() == 10);
        REQUIRE(digitisedNumbers[0]->rows() == 784);
    }

    SECTION("Read as matricies") 
    {
        string filename = "train_test.csv";
        shared_ptr<Matrix> labels;
        shared_ptr<Matrix> data;

        read_digit_data(path + filename, labels, data);

        REQUIRE(labels->cols() == data->cols());
        REQUIRE(labels->cols() == 301);
        REQUIRE(labels->rows() == 10);
        REQUIRE(data->rows() == 784);
    }

}