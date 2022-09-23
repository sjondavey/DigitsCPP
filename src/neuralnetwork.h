#ifndef NN_INCLUDED
#define NN_INCLUDED

#include <vector>
#include <functional>
#include "typedefs.h"
#include "readwritematrixtocsv.h"
#include "functions.h"
#include <iomanip> // for spoacing of cout

using namespace std;

namespace neuralnetworkfirstprinciples {

/**
 * An example using Gradient Descent to 'solve' for the digit recognition problem:
 *      https://www.kaggle.com/competitions/digit-recognizer
 * This is a vectorized implementation where the entire training set is loaded worked on
 * simultaneously. 
 * The solution makes use of the Eigen3 matrix library (https://eigen.tuxfamily.org/) 
 * which can run in multiple threads using OpenMP.
 */
class NeuralNetwork
{
    public:
    /* Inputs to the NN consist of
        - nodes_per_layer: In the 'digit recognition' problem the input layer consists of 784 (=28x28) nodes,
                           the output consists of 10 nodes and there may be one or more hidden layers. An example
                           of this vector would be (728, 200, 50, 10). This implementation is not intended to 
                           handle layers with many nodes. While I use 'unsigned short' as an attempt to indicate
                           this constraint, the top end of unsigned short is still way too large. 
        - learning_rate: In these examples a value of around 0.5 or 1.0 seems to work well
        - constants: for each transition between layers (i.e constants.size() == nodes_per_layer.size() - 1) this
                     is a column vector of constants which are needed to optimise gradient descent. Note in the 
                     literature, constants are referred to using the letter 'b'
        - weights: for each transition between layers (i.e weights.size() == nodes_per_layer.size() - 1) this is
                   a matrix which, when applied to the input layer, will create an output that is the size of 
                   the output layer i.e if nodes_per_layer[i] = n_{i} and nodes_per_layer[i+1] then
                   matrix[i] has has dimension (n_{i} x n_{i+1}). In the literature, weights are referred to using 
                   the letter 'W'

        Parameters (i.e constants and weights) can either be generated randomly or read from file using the functionality
        in functions.h (for random) or readwritematrixtocsv.h (for parameters read from disk)

    TODO: Confirm that constants and weights should be passed by reference. This may make sense if we are writing
          a function which needs to modify these Parameters in place but I don't think it should work like this 
          in a class object. A class should (?) make a copy of these input parameters so they cannot be messed with
    */
    NeuralNetwork(const vector<unsigned short> nodes_per_layer, 
                  const Scalar learning_rate, 
                  vector<shared_ptr<ColVector>>& constants, 
                  vector<shared_ptr<Matrix>>& weights);

    // Does not output anything but the Parameters (weights and constants) are updated
    void train(shared_ptr<Matrix>& data,
               shared_ptr<Matrix>& labels,
               size_t epochs,
               size_t output_cost_accuracy_every_n_steps = 100);

    // Returns, Y_hat = A[final_layer_number], the last calculation of the propagate_forward method
    shared_ptr<Matrix> evaluate(shared_ptr<Matrix>& data);


    // For testing only
    vector<unsigned short> get_nodes_per_layer();


    private:
    void set_input_data(shared_ptr<Matrix>& data);
    void set_expected_results(shared_ptr<Matrix>& labels);
    // Make sure to have set the input data before calling this method
    // If we set A[0] to input data, this method  
    // for i in (0, num_layers):
    //      Z[i] = A[i] \times W[i] + b[i];
    //      A[i+1] = activation_function(Z[i])
    // Indexing in this method can be a little confusing because, if we set A[0] to the input
    // data, it will be one element longer than every other vector
    void propagate_forward();
    void propagate_backward();
    void update_weights(); 



    vector<unsigned short> nodes_per_layer; 
    Scalar learning_rate;

    shared_ptr<Matrix> data;
    size_t number_of_training_examples;
    shared_ptr<Matrix> labels;
    unique_ptr<Matrix> ones; // for internal calculation during the back propagation step

    vector<shared_ptr<Matrix>> unactivated_values; // stores the un-activated (activation fn not yet applied) values of layers (i.e. Z = WX + b)
    vector<shared_ptr<Matrix>> d_unactivated_values; // derivative of unactivated_values
    vector<shared_ptr<Matrix>> neuron_values;      // stores the different layers of out network (AKA 'activated' vector A = sigmoid(Z))
    vector<shared_ptr<Matrix>> d_neuron_values;    // stores the error contribution of each neurons (i.e. dA where A = sigmoid(Z))
    vector<shared_ptr<Matrix>> weights;            // Weights itself (W)
    vector<shared_ptr<Matrix>> d_weights;          // derivative of the weights (dW)
    vector<shared_ptr<ColVector>> constants;       // Constants (b)
    vector<shared_ptr<ColVector>> d_constants;     // derivative of the constants 
};

}

#endif

/*
class NeuralNetwork
{
    public:
    // Set the weights and constants to random starting values
    NeuralNetwork(const vector<unsigned long> nodes_per_layer, const float learning_rate);
    // Use the input weights and constants - typically used to evaluate a test set.
    NeuralNetwork(const vector<unsigned long> nodes_per_layer, const float learning_rate, 
                  vector<shared_ptr<ColVector>>& constants, vector<shared_ptr<Matrix>>& weights);

    void set_output_directory(string output_directory);

    // the first layer in neuronLayers is set to 'input' and the subsequent neuronLayers are updated
    // leaving the estimated values in the final layer which can be queried using  the
    // final_layer_after_forward_propagation() method.
    void propagate_forward(ColVector& input);
    // function for backward propagation of errors made by neurons. 
    // 'output' = y^{[i]} or the i-th label from the training data for i = 1,...,m
    void propagate_backward(ColVector& output);

    void update_weights(size_t number_of_training_examples); // m = number_of_training_examples

    // writes the weights W and the constants b to a series of files 
    //     {parameter_file_prefix}_w0; ... ,{parameter_file_prefix}_wn; and
    //     {parameter_file_prefix}_b0; ... ,{parameter_file_prefix}_bn;
    // in the output_directory
    void write_parameters_to_file(string parameter_file_prefix);

    void train(vector<shared_ptr<ColVector>> training_data, 
               vector<shared_ptr<ColVector>> training_labels,
               size_t epochs);

    // For testing only
    vector<unsigned long> get_nodes_per_layer();
    shared_ptr<ColVector> final_layer_after_forward_propagation();


    private:
    string output_directory;
    void setup_topology(const vector<unsigned long> nodes_per_layer, const float learning_rate);
    vector<unsigned long> nodes_per_layer; 
    float learning_rate;

    vector<shared_ptr<ColVector>> neuronLayers; // stores the different layers of out network
    vector<shared_ptr<ColVector>> cacheLayers; // stores the un-activated (activation fn not yet applied) values of layers (i.e. Z = WX + b)
    vector<shared_ptr<ColVector>> deltas; // stores the error contribution of each neurons (i.e. dA where A = sigmoid(Z))
    vector<shared_ptr<Matrix>> weights; // the connection weights itself
    vector<shared_ptr<Matrix>> d_weights; // the deltas in the weights
    vector<shared_ptr<ColVector>> constants;
    vector<shared_ptr<ColVector>> d_constants;
};
*/