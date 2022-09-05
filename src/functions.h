#ifndef NN_FUNCTIONS_INCLUDED
#define NN_FUNCTIONS_INCLUDED

#include <vector>
#include <functional>
#include "typedefs.h"

using namespace std;

namespace neuralnetworkfirstprinciples {



////////////////////////////////////////////////////////////////////////////////
// TODO : Move this to some service class so it can be used in the Parallel version without having to include this class
// TODO : also include the get_cost_value() and get_accuracy() methods

// Just split into the first 'n' and the remaining "m-n'"
void split(vector<shared_ptr<ColVector>> full_data, 
           vector<shared_ptr<ColVector>> full_labels, 
           vector<shared_ptr<ColVector>>& training_data, 
           vector<shared_ptr<ColVector>>& training_labels, 
           vector<shared_ptr<ColVector>>& test_data, 
           vector<shared_ptr<ColVector>>& test_labels,
           size_t training_set_size);

void split(shared_ptr<Matrix> full_data, 
           shared_ptr<Matrix> full_labels, 
           shared_ptr<Matrix>& training_data, 
           shared_ptr<Matrix>& training_labels, 
           shared_ptr<Matrix>& test_data, 
           shared_ptr<Matrix>& test_labels,
           size_t training_set_size);

// Only Sigmoid function for now
Scalar activation_function(Scalar x);

Scalar activation_function_derivative(Scalar x);

// Logistic regression cost function
Scalar get_cost_value(shared_ptr<ColVector> y_hat, shared_ptr<ColVector> y);
Scalar get_cost_value(unique_ptr<ColVector>& y_hat, shared_ptr<ColVector> y);
Scalar get_cost_value(shared_ptr<Matrix> Y_hat, shared_ptr<Matrix> Y);

// converts y_hat into a column vector os 0s and 1s and compares this to the 0 and 1 values in y. Returns 1 if 
// the converted y_hat matches y at every entry
int get_accuracy(shared_ptr<ColVector> y_hat, shared_ptr<ColVector> y);
int get_accuracy(unique_ptr<ColVector>& y_hat, shared_ptr<ColVector> y);
Scalar get_accuracy(shared_ptr<Matrix> Y_hat, shared_ptr<Matrix> Y);

void generate_random_weights(const vector<unsigned short> nodes_per_layer,
                             vector<shared_ptr<Matrix>>& weights,
                             vector<shared_ptr<ColVector>>& constants);

void generate_random_weights(const vector<unsigned short> nodes_per_layer,
                             vector<shared_ptr<Matrix>>& weights,
                             vector<shared_ptr<ColVector>>& constants);

////////////////////////////////////////////////////////////////////////////////



}

#endif