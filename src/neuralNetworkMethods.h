#ifndef NN_METHODS_INCLUDED
#define NN_METHODS_INCLUDED

#include <vector>
#include <functional>
#include "typedefs.h"
#include "readwritematrixtocsv.h"
#include "functions.h"
#include "readwritematrixtocsv.h"
#include <iomanip> // for spoacing of cout
#include <ppl.h>

//////////////////////////////////////////////////////////////////////////////////////////////
// #define EIGEN_DONT_PARALLELIZE
//////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

namespace neuralnetworkfirstprinciples {

    // All the 'train' methods work by modifing the intput constants and weights which can then be used
    // by the 'test' methods
    void train_vectorized(const vector<unsigned short> nodes_per_layer, const float learning_rate, 
                          vector<shared_ptr<ColVector>>& constants, vector<shared_ptr<Matrix>>& weights,
                          shared_ptr<Matrix> data, shared_ptr<Matrix> labels,
                          size_t epochs,
                          size_t output_cost_accuracy_every_n_steps = 50);

    shared_ptr<Matrix> test_vectorized(vector<shared_ptr<ColVector>>& constants, vector<shared_ptr<Matrix>>& weights,
                        shared_ptr<Matrix> data, shared_ptr<Matrix> labels);

    void train_loop_faster(const vector<unsigned short> nodes_per_layer, const float learning_rate, 
                          vector<shared_ptr<ColVector>>& constants, vector<shared_ptr<Matrix>>& weights,
                          vector<shared_ptr<ColVector>>& data, vector<shared_ptr<ColVector>>& labels,
                          size_t epochs,
                          size_t output_cost_accuracy_every_n_steps = 50);
                          

    void train_loop_base(const vector<unsigned short> nodes_per_layer, const float learning_rate, 
                          vector<shared_ptr<ColVector>>& constants, vector<shared_ptr<Matrix>>& weights,
                          vector<shared_ptr<ColVector>>& data, vector<shared_ptr<ColVector>>& labels,
                          size_t epochs, 
                          size_t output_cost_accuracy_every_n_steps = 50);

}

#endif