#include "catch.hpp"
#include "../src/functions.h"
#include <filesystem>

using namespace std;
using namespace neuralnetworkfirstprinciples;

void create_input_data(shared_ptr<Matrix>& input_data,
                       shared_ptr<Matrix>& input_labels)
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
}


TEST_CASE("Test the neural network support functions", "[supportfunctions]") {

    string path = "E:/Code/kaggle/digits/test/";

    SECTION("Split data")
    {
        shared_ptr<Matrix> input_data, input_labels;
        create_input_data(input_data, input_labels);

        shared_ptr<Matrix> training_data, training_labels, test_data, test_labels;
        size_t training_set_size = 1;
        split(input_data, input_labels,
              training_data, training_labels,
              test_data, test_labels, 
              training_set_size);
        REQUIRE(training_data->cols() == 1);
        REQUIRE(training_data->rows() == 3);
        REQUIRE(abs((*training_data)(1,0) - 0.11) < 1e-6);

        REQUIRE(training_labels->cols() == 1);
        REQUIRE(training_labels->rows() == 2);
        REQUIRE(abs((*training_labels)(1,0) - 0.536085) < 1e-6);

        REQUIRE(test_data->cols() == 1);
        REQUIRE(test_data->rows() == 3);
        REQUIRE(abs((*test_data)(1,0) - 0.22) < 1e-6);

        REQUIRE(test_labels->cols() == 1);
        REQUIRE(test_labels->rows() == 2);
        REQUIRE(abs((*test_labels)(1,0) - 0.482733) < 1e-6);
    }

    SECTION("Base Accuracy Function")
    {
        shared_ptr<ColVector> y_hat= make_shared<ColVector>(3, 1);
        (*y_hat)(0) = 0.9;
        (*y_hat)(1) = 0.1;
        (*y_hat)(2) = 0.7;

        shared_ptr<ColVector> y = make_shared<ColVector>(3, 1);
        (*y)(0) = 1;
        (*y)(1) = 0;
        (*y)(2) = 0;
        REQUIRE(get_accuracy(y_hat, y) == 0);

        (*y_hat)(2) = 0.4;
        REQUIRE(get_accuracy(y_hat, y) == 1);
    }
    
    SECTION("Vectorized Accuracy Function") 
    {
        shared_ptr<Matrix> Y_hat = make_shared<Matrix>(3,2);
        (*Y_hat)(0,0) = 0.9;
        (*Y_hat)(1,0) = 0.1;
        (*Y_hat)(2,0) = 0.7;
        (*Y_hat)(0,1) = 0.9;
        (*Y_hat)(1,1) = 0.1;
        (*Y_hat)(2,1) = 0.4;

        shared_ptr<Matrix> Y = make_shared<Matrix>(3,2);
        (*Y)(0,0) = 1;
        (*Y)(1,0) = 0;
        (*Y)(2,0) = 0;
        (*Y)(0,1) = 1;
        (*Y)(1,1) = 0;
        (*Y)(2,1) = 0;

        REQUIRE(abs(get_accuracy(Y_hat, Y) - 0.5) < 1e-6);

        (*Y_hat)(2,0) = 0.4;
        (*Y_hat)(1,1) = 0.8;
        REQUIRE(abs(get_accuracy(Y_hat, Y) - 0.5) < 1e-6);

        (*Y_hat)(1,1) = 0.2;
        REQUIRE(abs(get_accuracy(Y_hat, Y) - 1.0) < 1e-6);

        // BUG Case, All estimates are below 0.5
        (*Y_hat)(0,0) = 0.4;
        (*Y_hat)(2,0) = 0.4;
        REQUIRE(abs(get_accuracy(Y_hat, Y) - 0.5) < 1e-6);
    }

    SECTION("Base Cost Function")
    {
        // Check get_cost_value() - NOTE: Just the method, not using vectors that are the correct size according to the NN
        shared_ptr<ColVector> more_constants = make_shared<ColVector>(3, 1);
        (*more_constants)(0) = 0.11;
        (*more_constants)(1) = 0.22;
        (*more_constants)(2) = 0.33;

        shared_ptr<ColVector> test_constants = make_shared<ColVector>(3, 1);
        (*test_constants)(0) = 0.1;
        (*test_constants)(1) = 0.2;
        (*test_constants)(2) = 0.3;

        Scalar manual_cost = 0;
        for (size_t i = 0; i < 3; ++i)
        {
            manual_cost += (*more_constants)(i) * log((*test_constants)(i)) + (1-(*more_constants)(i)) * log(1-(*test_constants)(i));
        }
        manual_cost *= -1;
        Scalar calculated_cost = get_cost_value(test_constants, more_constants);
        REQUIRE(abs(manual_cost - calculated_cost) < 1e-8);
    }

}