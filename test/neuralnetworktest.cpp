#include "catch.hpp"
#include "../src/neuralnetwork.h"
#include "../src/inputdatafilereader.h"

using namespace std;
using namespace neuralnetworkfirstprinciples;

// Some test data
void create_input_data(shared_ptr<Matrix>& input_data,
                       shared_ptr<Matrix>& input_labels,
                       vector<shared_ptr<ColVector>> &test_constants,
                       vector<shared_ptr<Matrix>> &test_weights)
{
    input_data = make_shared<Matrix>(3,2); // i.e. two examples each with 3 labels
    (*input_data)(0,0) = 0.1;
    (*input_data)(1,0) = 0.11;
    (*input_data)(2,0) = 0.12;

    (*input_data)(0,1) = 0.11;
    (*input_data)(1,1) = 0.22;
    (*input_data)(2,1) = 0.33;

    input_labels = make_shared<Matrix>(2,2);
    (*input_labels)(0,0) = 0.501135;
    (*input_labels)(1,0) = 0.536085;

    (*input_labels)(0,1) = 0.451226;
    (*input_labels)(1,1) = 0.482733;


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

TEST_CASE("Neural Network implementation", "[neuralnetworktest]") {



    SECTION("Neural Network")
    {
        shared_ptr<Matrix> input_data, input_labels;
        vector<shared_ptr<ColVector>> test_constants;
        vector<shared_ptr<Matrix>> test_weights;

        create_input_data(input_data, input_labels, test_constants, test_weights);
        vector<unsigned short> nodes_per_layer{3,3,2};
        float learning_rate = 0.5;

        NeuralNetwork nn2 = NeuralNetwork(nodes_per_layer, learning_rate, test_constants, test_weights);
        vector<unsigned short> nodes;
        nodes = nn2.get_nodes_per_layer();
        REQUIRE(nodes.size() == nodes_per_layer.size());
        REQUIRE(nodes[0] == nodes_per_layer[0]);

        // Forward propagation against manual matrix multiplication
        shared_ptr<Matrix> pseudo_start = input_data;
        shared_ptr<Matrix> manual_result = make_shared<Matrix>(3,2);
        (*manual_result) = (*test_weights[0]) * (*pseudo_start);
        (*manual_result).colwise() += (*test_constants[0]);
        (*manual_result) = (*manual_result).unaryExpr(&activation_function);
        //step 2
        (*manual_result) = (*test_weights[1]) * (*manual_result);
        (*manual_result).colwise() +=  (*test_constants[1]);
        (*manual_result) = (*manual_result).unaryExpr(&activation_function);

        shared_ptr<Matrix> code_result = nn2.evaluate(input_data);
        REQUIRE(code_result->rows() == 2);
        REQUIRE(code_result->cols() == 2);
        REQUIRE(abs((*code_result)(0,0) - (*manual_result)(0,0)) < 1e-12);

        // back propagation
        size_t epochs = 1;
        shared_ptr<Matrix> output = make_shared<Matrix>((*manual_result) * 0.9);
        nn2.train(input_data, output, epochs);

        // Write some tests here for the private variables. Perhaps forward propagate and check?
        shared_ptr<Matrix> second_training_example = make_shared<Matrix>(3,2);
        (*second_training_example)(0,0) = 0.1;
        (*second_training_example)(1,0) = 0.11;
        (*second_training_example)(2,0) = 0.12;

        (*second_training_example)(0,1) = 0.12;
        (*second_training_example)(1,1) = 0.13;
        (*second_training_example)(2,1) = 0.14;

        shared_ptr<Matrix> second_output =  make_shared<Matrix>(*nn2.evaluate(second_training_example) * 0.9);


        // cout << "Value 1: " << (*second_output)(0,1) << endl;
        // cout << "Value 2: " <<  (*second_output)(1,1) << endl;
        REQUIRE( abs((*second_output)(0,1) - 0.489046) < 1e-6);
        REQUIRE( abs((*second_output)(1,1) - 0.52346) < 1e-6);
    }

    SECTION("Regression Test") 
    {
        shared_ptr<Matrix> labels;
        shared_ptr<Matrix> data;
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

        NeuralNetwork nn = NeuralNetwork(nodes_per_layer, learning_rate, constants, weights);
        nn.train(data, labels, epochs);
        // write_parameters(weights, constants, path, "tmp_test");

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

