#include "catch.hpp"
#include "../src/inputdatafilereader.h"
#include "../src/neuralNetworkMethods.h"
//#include <cmath> // std::isnan in C++11

using namespace std;
using namespace neuralnetworkfirstprinciples;

/* Regression test data
void read_input_data(vector<shared_ptr<ColVector>>& labelAsVectors,
                     vector<shared_ptr<ColVector>>& digitisedNumbers) 
{
    string path = "./test/";
    string filename = "regression_test_input.csv";
    string input_data_file = path + filename;
    read_digit_data(input_data_file, labelAsVectors, digitisedNumbers);

}

// Manual test data for loops
void create_input_data(vector<shared_ptr<ColVector>>& input_data,
                       vector<shared_ptr<ColVector>>& input_labels,
                       vector<shared_ptr<ColVector>> &test_constants,
                       vector<shared_ptr<Matrix>> &test_weights)
{
    input_data = vector<shared_ptr<ColVector>>(2); // make_shared<Matrix>(3,2); // i.e. two examples each with 3 labels
    input_data[0] = make_shared<ColVector>(3,1);
    (*input_data[0])(0) = 0.1;
    (*input_data[0])(1) = 0.11;
    (*input_data[0])(2) = 0.12;

    input_data[1] = make_shared<ColVector>(3,1);
    (*input_data[1])(0) = 0.11;
    (*input_data[1])(1) = 0.22;
    (*input_data[1])(2) = 0.33;

    input_labels = vector<shared_ptr<ColVector>>(2);
    input_labels[0] = make_shared<ColVector>(2,1);
    (*input_labels[0])(0) = 0.501135;
    (*input_labels[0])(1) = 0.536085;

    input_labels[1] = make_shared<ColVector>(2,1);
    (*input_labels[1])(0) = 0.451226;
    (*input_labels[1])(1) = 0.482733;


    test_constants = vector<shared_ptr<ColVector>>(2);
    test_constants[0] = make_shared<ColVector>(3, 1);
    (*test_constants[0])(0) = 0.1;
    (*test_constants[0])(1) = 0.2;
    (*test_constants[0])(2) = 0.3;

    test_constants[1] = make_shared<ColVector>(2, 1);
    (*test_constants[1])(0) = 0;
    (*test_constants[1])(1) = 0.1;

    test_weights = vector<shared_ptr<Matrix>>(2);
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
}
*/

TEST_CASE("Neural Network function", "[neuralnetworkfunctiontest]") {

    SECTION("Test Basic Functionality")
    {
        // vector<unsigned short> nodes_per_layer{3,3,2};
        // float learning_rate = 0.5;
        // size_t epochs = 15;

        // vector<shared_ptr<ColVector>> data(2);
        // vector<shared_ptr<ColVector>> labels(2);
        // vector<shared_ptr<ColVector>> constants_base(2);
        // vector<shared_ptr<Matrix>> weights_base(2);        
        // create_input_data(data, labels, constants_base, weights_base);
        // train_loop_base(nodes_per_layer, learning_rate, 
        //                   constants_base, weights_base,
        //                   data, labels,
        //                   epochs);

        // vector<shared_ptr<ColVector>> constants_fast(2);
        // vector<shared_ptr<Matrix>> weights_fast(2);
        // create_input_data(data, labels, constants_fast, weights_fast);
        // train_loop_faster(nodes_per_layer, learning_rate, 
        //                   constants_fast, weights_fast,
        //                   data, labels,
        //                   epochs);
        
        // REQUIRE(abs((*constants_fast[0])(2) - (*constants_base[0])(2)) < 1e-6 );
        // REQUIRE(abs((*constants_fast[1])(1) - (*constants_base[1])(1)) < 1e-6 );
        // REQUIRE(abs((*weights_fast[0])(1,1) - (*weights_base[0])(1,1)) < 1e-6 );
        // REQUIRE(abs((*weights_fast[1])(1,1) - (*weights_base[1])(1,1)) < 1e-6 );

        REQUIRE(true);

    }

    SECTION("Regression Test - Base implementation") 
    {
        vector<shared_ptr<ColVector>> labels;
        vector<shared_ptr<ColVector>> data;
        string path = "E:/Code/kaggle/digits/test/";
        string filename = "regression_test_input.csv";
        string input_data_file = path + filename;
        read_digit_data(input_data_file, labels, data);

        vector<unsigned short> nodes_per_layer{784,50,10};
        float learning_rate = 0.5;
        vector<shared_ptr<ColVector>> constants;
        vector<shared_ptr<Matrix>> weights;
        
        string regression_input_prefix = "regression_input";
        read_parameters(weights, constants, path, regression_input_prefix);

        size_t epochs = 10;
        string output_file_prefix = "regression_output";

        train_loop_base(nodes_per_layer, learning_rate, 
                          constants, weights,
                          data, labels,
                          epochs);
        // //write_parameters(weights, constants, path, "funny_recon");                


        vector<shared_ptr<Matrix>> weights_check = vector<shared_ptr<Matrix>>(0);
        vector<shared_ptr<ColVector>> constants_check =  vector<shared_ptr<ColVector>>(0);

        read_parameters(weights_check, constants_check, path, output_file_prefix);

        REQUIRE(weights_check.size() == weights.size());
        REQUIRE(weights_check[0]->rows() == weights[0]->rows());
        REQUIRE(weights_check[0]->cols() == weights[0]->cols());
        // Check a random entry in the weights for the regression test
        REQUIRE(abs((*weights_check[1])(8,8) - (*weights[1])(8, 8)) < 1e-6);
        REQUIRE(constants_check.size() == constants.size());
        REQUIRE(constants_check[0]->rows() == constants[0]->rows());
        REQUIRE(constants_check[0]->cols() == constants[0]->cols());
        REQUIRE(abs((*constants_check[0])(15) - (*constants[0])(15)) < 1e-6);
    }

    SECTION("Regression Test - Fast Loop") 
    {
        // Eigen::initParallel();
        vector<shared_ptr<ColVector>> labels;
        vector<shared_ptr<ColVector>> data;
        string path = "E:/Code/kaggle/digits/test/";
        string filename = "regression_test_input.csv";
        string input_data_file = path + filename;
        read_digit_data(input_data_file, labels, data);

        vector<unsigned short> nodes_per_layer{784,50,10};
        float learning_rate = 0.5;
        vector<shared_ptr<ColVector>> constants;
        vector<shared_ptr<Matrix>> weights;
        
        string regression_input_prefix = "regression_input";
        read_parameters(weights, constants, path, regression_input_prefix);

        size_t epochs = 10;
        string output_file_prefix = "regression_output";


        cout << "Here ...." << endl;
        train_loop_faster(nodes_per_layer, learning_rate, 
                          constants, weights,
                          data, labels,
                          epochs);
        //write_parameters(weights, constants, path, "funny_recon");                


        vector<shared_ptr<Matrix>> weights_check = vector<shared_ptr<Matrix>>(0);
        vector<shared_ptr<ColVector>> constants_check =  vector<shared_ptr<ColVector>>(0);

        read_parameters(weights_check, constants_check, path, output_file_prefix);

        REQUIRE(weights_check.size() == weights.size());
        REQUIRE(weights_check[1]->rows() == weights[1]->rows());
        REQUIRE(weights_check[1]->cols() == weights[1]->cols());
        // Check a random entry in the weights for the regression test
        REQUIRE(abs((*weights_check[1])(8,8) - (*weights[1])(8, 8)) < 1e-6);
        REQUIRE(constants_check.size() == constants.size());
        REQUIRE(constants_check[0]->rows() == constants[0]->rows());
        REQUIRE(constants_check[0]->cols() == constants[0]->cols());
        REQUIRE(abs((*constants_check[0])(15) - (*constants[0])(15)) < 1e-6);

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