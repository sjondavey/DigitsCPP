#include "catch.hpp"
#include "../src/inputdatafilereader.h"
#include "../src/neuralNetworkMethods.h"
//#include <cmath> // std::isnan in C++11

using namespace std;
using namespace neuralnetworkfirstprinciples;

// Some test data
void read_input_data(vector<shared_ptr<ColVector>>& labelAsVectors,
                     vector<shared_ptr<ColVector>>& digitisedNumbers) 
{
    string path = "./test/";
    string filename = "regression_test_input.csv";
    string input_data_file = path + filename;
    read_digit_data(input_data_file, labelAsVectors, digitisedNumbers);

}

TEST_CASE("Neural Network function", "[neuralnetworkfunctiontest]") {


    SECTION("Regression Test") 
    {
        // vector<shared_ptr<ColVector>> labels;
        // vector<shared_ptr<ColVector>> data;
        // string path = "E:/Code/kaggle/digits/test/";
        // string filename = "regression_test_input.csv";
        // string input_data_file = path + filename;
        // read_digit_data(input_data_file, labels, data);

        // vector<unsigned short> nodes_per_layer{784,50,10};
        // float learning_rate = 0.5;
        // vector<shared_ptr<ColVector>> constants;
        // vector<shared_ptr<Matrix>> weights;
        
        // // Eigen3 uses the system seed so should always generate the same starting weights    
        // // std::srand(42);    
        // // generate_random_weights(nodes_per_layer, weights, constants);
        // string regression_input_prefix = "regression_input";
        // readParameters(weights, constants, path, regression_input_prefix);

        // size_t epochs = 10;
        // string output_file_prefix = "regression_output";

        // one_method_train(nodes_per_layer, learning_rate, 
        //                 constants, weights,
        //                 data, labels,
        //                 epochs);
        // writeParameters(weights, constants, path, "funny_recon");                


        // vector<shared_ptr<Matrix>> weights_check = vector<shared_ptr<Matrix>>(0);
        // vector<shared_ptr<ColVector>> constants_check =  vector<shared_ptr<ColVector>>(0);

        // readParameters(weights_check, constants_check, path, output_file_prefix);

        // REQUIRE(weights_check.size() == weights.size());
        // REQUIRE(weights_check[0]->rows() == weights[0]->rows());
        // REQUIRE(weights_check[0]->cols() == weights[0]->cols());
        // // Check a random entry in the weights for the regression test
        // REQUIRE(abs((*weights_check[0])(10,10) - (*weights[0])(10, 10)) < 1e-6);
        // REQUIRE(constants_check.size() == constants.size());
        // REQUIRE(constants_check[0]->rows() == constants[0]->rows());
        // REQUIRE(constants_check[0]->cols() == constants[0]->cols());
        // REQUIRE(abs((*constants_check[0])(15) - (*constants[0])(15)) < 1e-6);

        REQUIRE(true);
    }

    SECTION("Vectorized implementation") 
    {
        shared_ptr<Matrix> labels;
        shared_ptr<Matrix> data;
        string path = "E:/Code/kaggle/digits/test/";
        string filename = "regression_test_input.csv";
        string parameters_prefix = "regression_output";
        string input_data_file = path + filename;
        read_digit_data(input_data_file, labels, data);

        vector<unsigned short> nodes_per_layer{784,50,10};
        float learning_rate = 0.5;
        vector<shared_ptr<ColVector>> constants;
        vector<shared_ptr<Matrix>> weights;

        /*
        readParameters(weights, constants, path, parameters_prefix);
        size_t epochs = 10;
        */
        
        string regression_input_prefix = "regression_input";
        read_parameters(weights, constants, path, regression_input_prefix);

        size_t epochs = 10;
        string output_file_prefix = "regression_output";
        

        train_vectorized(nodes_per_layer, learning_rate, 
                        constants, weights,
                        data, labels,
                        epochs);

        // writeParameters(weights, constants, path, output_file_prefix);


        vector<shared_ptr<Matrix>> weights_check = vector<shared_ptr<Matrix>>(0);
        vector<shared_ptr<ColVector>> constants_check =  vector<shared_ptr<ColVector>>(0);

        read_parameters(weights_check, constants_check, path, output_file_prefix);

        REQUIRE(weights_check.size() == weights.size());
        REQUIRE(weights_check[0]->rows() == weights[0]->rows());
        REQUIRE(weights_check[0]->cols() == weights[0]->cols());
        // Check a random entry in the weights for the regression test
        REQUIRE(abs((*weights_check[0])(10,10) - (*weights[0])(10, 10)) < 1e-6);
        REQUIRE(constants_check.size() == constants.size());
        REQUIRE(constants_check[0]->rows() == constants[0]->rows());
        REQUIRE(constants_check[0]->cols() == constants[0]->cols());
        REQUIRE(abs((*constants_check[0])(15) - (*constants[0])(15)) < 1e-6);

    }

}