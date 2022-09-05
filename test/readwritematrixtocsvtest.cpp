#include "catch.hpp"
#include "../src/readwritematrixtocsv.h"
#include <filesystem>

using namespace std;
using namespace neuralnetworkfirstprinciples;

TEST_CASE("Neural Network parameters to disk", "[readwriteparameterstest]") {

    SECTION("Check read and write or parameters") 
    {
        vector<shared_ptr<ColVector>> test_constants = vector<shared_ptr<ColVector>>(2);
        test_constants[0] = make_shared<ColVector>(3, 1);
        (*test_constants[0])(0) = 0.1;
        (*test_constants[0])(1) = 0.2;
        (*test_constants[0])(2) = 0.3;

        test_constants[1] = make_shared<ColVector>(2, 1);
        (*test_constants[1])(0) = 0;
        (*test_constants[1])(1) = 0.1;

        vector<shared_ptr<Matrix>> test_weights = vector<shared_ptr<Matrix>>(2);
        test_weights[0] = make_shared<Matrix>(3,3);
        (*test_weights[0])(0,0) = 0.1;
        (*test_weights[0])(0,1) = 0.1;
        (*test_weights[0])(0,2) = 0.2;
        (*test_weights[0])(1,0) = 0.1;
        (*test_weights[0])(1,1) = 0.1;
        (*test_weights[0])(1,2) = 0.3;
        (*test_weights[0])(2,0) = 0.1;
        (*test_weights[0])(2,1) = 0.1;
        (*test_weights[0])(2,2) = 0.4;

        test_weights[1] = make_shared<Matrix>(2,3);
        (*test_weights[1])(0,0) = 0.1;
        (*test_weights[1])(0,1) = 0.1;
        (*test_weights[1])(0,2) = 0.2;
        (*test_weights[1])(1,0) = 0.1;
        (*test_weights[1])(1,1) = 0.1;
        (*test_weights[1])(1,2) = 0.3;

        string path = "E:/Code/kaggle/digits/data/";
        string file_name_prefix = "test_parameters";
        write_parameters(test_weights, test_constants, path, file_name_prefix);


        string weights_file_name = path + file_name_prefix + "_W0.csv";
        ifstream f = ifstream(weights_file_name.c_str());
        REQUIRE(f.good());
        string constants_file_name = path + file_name_prefix + "_b0.csv";
        f = ifstream(constants_file_name.c_str());
        REQUIRE(f.good());

        weights_file_name = path + file_name_prefix + "_W1.csv";
        f = ifstream(weights_file_name.c_str());
        REQUIRE(f.good());
        constants_file_name = path + file_name_prefix + "_b1.csv";
        f = ifstream(constants_file_name.c_str());
        REQUIRE(f.good());

        vector<shared_ptr<Matrix>> read_weights;
        vector<shared_ptr<ColVector>> read_constants;
        read_parameters(read_weights, read_constants, path, file_name_prefix);
        REQUIRE(read_weights.size() == 2);
        REQUIRE(read_constants.size() == 2);

        REQUIRE(read_weights[0]->rows() == 3);
        REQUIRE(read_weights[0]->cols() == 3);
        REQUIRE(read_constants[1]->rows() == 2);
        REQUIRE(read_constants[1]->cols() == 1);

        REQUIRE((*read_weights[0])(2,1) == (*test_weights[0])(2,1));
        REQUIRE((*read_constants[1])(1) == (*test_constants[1])(1));


        try {
            std::filesystem::remove(path + file_name_prefix + "_w0.csv");
            std::filesystem::remove(path + file_name_prefix + "_w1.csv");
            std::filesystem::remove(path + file_name_prefix + "_b0.csv");
            string pathname = path + file_name_prefix + "_b1.csv";
            f = ifstream(pathname);
            REQUIRE(f.good());
            bool ok = std::filesystem::remove(pathname);
            REQUIRE(ok);
        }
        catch(const std::filesystem::filesystem_error& err) {
            std::cout << "filesystem error: " << err.what() << '\n';
        }

    }
}