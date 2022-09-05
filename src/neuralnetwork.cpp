#include "neuralnetwork.h"

using namespace std;

namespace neuralnetworkfirstprinciples {

    NeuralNetwork::NeuralNetwork(const vector<unsigned short> nodes_per_layer, 
                                 const float learning_rate, 
                                 vector<shared_ptr<ColVector>>& constants, 
                                 vector<shared_ptr<Matrix>>& weights)
    {
        this->nodes_per_layer = nodes_per_layer;
        this->learning_rate = learning_rate;
        // Check input dimensions and set 
        if (constants.size() != nodes_per_layer.size() - 1)
            throw length_error("Input constants do not have the correct number of column vectors");
        if (weights.size() != nodes_per_layer.size() - 1)
            throw length_error("Input weights do not have the correct number of matrices");

        for (size_t i = 1; i < nodes_per_layer.size(); i++) {
            if ((constants[i-1]->rows() != nodes_per_layer[i]) || (constants[i-1]->cols() != 1))
                throw length_error("Input constants do not have the correct dimensions");

            if ((weights[i-1]->rows() != nodes_per_layer[i]) || (weights[i-1]->cols() != nodes_per_layer[i-1]))
                throw length_error("Input weights do not have the correct dimensions");                
        }
        this->constants = constants;
        this->weights = weights;
    }

    void NeuralNetwork::set_input_data(shared_ptr<Matrix>& data)
    {
        this->data = data;
        number_of_training_examples = data->cols();

        // Indexing is always going to be an issue. There is a difference between the starting layer 
        // of unchanging input data, and all the subsequent layers which are a function of the parmaters
        // (weights and constants) that need to be updated every epoch
        //
        // TODO: Think about making all these vectors of length "nodes_per_layer.size()-1" and leaving
        // the initial "neuron_values[0]" layer out and only referring to it as 'input_data'. This 
        // may make the loop look a little odd (the first step from 'input_data' to neuron_layer[0] 
        // will have to happen outside of the loop which takes neuron_layer[n] to neuron_layer[n+1] but
        // the flip side is that all these internal vectors will have a more natural size and it should
        // be easier to read and understand the code)
        //
        // TODO: Irrespective of the above todo, d_neuron_values must be of size "nodes_per_layer.size()-1"
        // As we never need to worry about changes to the input data which is currently living in 
        // neuron_values[0]
        unactivated_values = vector<shared_ptr<Matrix>>(nodes_per_layer.size()-1); // Does not need to store anything relating to the input layer
        neuron_values =  vector<shared_ptr<Matrix>>(nodes_per_layer.size()); // Input data is stored in neuron_values[0] so this is one element longer

        // We may or may not be training here but the following structures, only needed for gradient 
        // descent are small so even if we are not training, there is not much overhead.
        //
        // TODO: Consider moving these so they are only created if we are training.
        d_unactivated_values = vector<shared_ptr<Matrix>>(nodes_per_layer.size()-1);
        d_neuron_values = vector<shared_ptr<Matrix>>(nodes_per_layer.size()-1);
        d_weights =       vector<shared_ptr<Matrix>>(nodes_per_layer.size() - 1);
        d_constants =     vector<shared_ptr<ColVector>>(nodes_per_layer.size() - 1);
        for (size_t i = 0; i < nodes_per_layer.size(); i++) {
            neuron_values[i] = make_shared<Matrix>(nodes_per_layer[i], number_of_training_examples);
            if (i > 0) {
                d_neuron_values[i-1] = make_shared<Matrix>(nodes_per_layer[i-1], number_of_training_examples);
                unactivated_values[i-1] = make_shared<Matrix>(nodes_per_layer[i-1], number_of_training_examples);
                d_unactivated_values[i-1] = make_shared<Matrix>(nodes_per_layer[i-1], number_of_training_examples);
                d_constants[i-1] = make_shared<ColVector>(nodes_per_layer[i-1]);
                d_constants[i-1]->setZero();
                d_weights[i-1] = make_shared<Matrix>(weights[i-1]->rows(), weights[i-1]->cols());
                d_weights[i-1]->setZero();
            }
        }
    }

    void NeuralNetwork::set_expected_results(shared_ptr<Matrix>& labels)
    {
        this->labels = labels;
        ones = make_unique<Matrix>(labels->rows(), labels->cols());
        ones->setOnes();
    }

    void  NeuralNetwork::propagate_forward()
    {
        neuron_values[0] = data;
        size_t last_layer = nodes_per_layer.size() - 1;

        for (size_t i = 0; i < last_layer; ++i)
        {
            *unactivated_values[i] = (*weights[i]) * (*neuron_values[i]);
            (*unactivated_values[i]).colwise() += (*constants[i]); // Adds the constants vector to each column of the unactivated_values matrix
            (*neuron_values[i+1]) = (*unactivated_values[i]).unaryExpr(&activation_function);
        }
    }

    shared_ptr<Matrix> NeuralNetwork::evaluate(shared_ptr<Matrix>& data)
    {
        set_input_data(data);
        propagate_forward();
        return (neuron_values.back());
    }


    void NeuralNetwork::propagate_backward()
    {
        size_t last_layer = nodes_per_layer.size() - 1;
        // string path = "E:/Code/kaggle/digits/test/";
        // string filename = path + "neuron_values_" + to_string(last_layer) + ".csv";
        // write_matrix_to_file(filename, *neuron_values[last_layer]);
        (*d_neuron_values[last_layer-1]) = -((*labels).cwiseQuotient(*neuron_values[last_layer]) - 
                                             ((*ones) - (*labels)).cwiseQuotient((*ones) - (*neuron_values[last_layer])) );
        // filename = path + "d_neuron_values_" + to_string(last_layer-1) + ".csv";
        // write_matrix_to_file(filename, *d_neuron_values[last_layer-1]);
        for (int i = last_layer-1; i >= 0; --i)
        {
            (*d_unactivated_values[i]) = (*d_neuron_values[i]).array() * ((*unactivated_values[i]).unaryExpr(&activation_function_derivative)).array();
                // string filename = path + "d_unactivated_values_" + to_string(i) + ".csv";
                // write_matrix_to_file(filename, *d_unactivated_values[i]);
            
            (*d_weights[i]) = ((*d_unactivated_values[i]) * (*neuron_values[i]).transpose()) / (Scalar) number_of_training_examples;
                // filename = path + "d_weights_" + to_string(i) + ".csv";
                // write_matrix_to_file(filename, *d_weights[i]);
            
            (*d_constants[i]) = (*d_unactivated_values[i]).rowwise().sum() / (Scalar) number_of_training_examples;
                // filename = path + "d_constants_" + to_string(i) + ".csv";
                // write_matrix_to_file(filename, *d_constants[i]);
            if (i > 0) 
            {
                (*d_neuron_values[i-1]) = (*weights[i]).transpose() * (*d_unactivated_values[i]);
                    // filename = path + "d_neuron_values_" + to_string(i-1) + ".csv";
                    // write_matrix_to_file(filename, *d_neuron_values[i-1]);            
            }
        }
    }

    void NeuralNetwork::update_weights()
    {
        for (int i = nodes_per_layer.size() - 2; i >= 0; --i)
        {
            (*weights[i]) -= learning_rate * (*d_weights[i]);
            d_weights[i]->setZero();
            (*constants[i]) -= learning_rate * (*d_constants[i]);
            d_constants[i]->setZero();
        }
    }

    void NeuralNetwork::train(shared_ptr<Matrix>& data,
                              shared_ptr<Matrix>& labels,
                              size_t epochs,
                              size_t output_cost_accuracy_every_n_steps)
    {
        set_input_data(data);
        set_expected_results(labels);

        for (size_t i = 0; i < epochs; ++i)
        {
            Scalar cost = 0;
            Scalar accuracy = 0;
            propagate_forward();
            propagate_backward();
            update_weights();

            if (i % output_cost_accuracy_every_n_steps == 0)
            {
                Scalar cost = get_cost_value((neuron_values.back()), labels);
                Scalar accuracy = (Scalar) get_accuracy((neuron_values.back()), labels);
                cout << "Epochs: " << std::setfill('0') << std::setw(5) 
                     << to_string(i) << ", cost: " << to_string(cost) << ", accuracy: " << to_string(accuracy) << endl;
            }
        }
    }


 
    vector<unsigned short> NeuralNetwork::get_nodes_per_layer()
    {
        return nodes_per_layer;
    }

}


/*
    NeuralNetwork::NeuralNetwork(const vector<unsigned long> nodes_per_layer, const float learning_rate)
    {
        setup_topology(nodes_per_layer, learning_rate);

        // initialize weights and constants matrix
         generate_random_weights(nodes_per_layer, weights, constants);
    }

    NeuralNetwork::NeuralNetwork(const vector<unsigned long> nodes_per_layer, 
                                 const float learning_rate, 
                                 vector<shared_ptr<ColVector>>& constants, 
                                 vector<shared_ptr<Matrix>>& weights)
    {
        setup_topology(nodes_per_layer, learning_rate);
        // Check input dimensions and set 
        if (constants.size() != nodes_per_layer.size() - 1)
            throw length_error("Input constants do not have the correct number of column vectors");
        if (weights.size() != nodes_per_layer.size() - 1)
            throw length_error("Input weights do not have the correct number of matrices");

        for (size_t i = 1; i < nodes_per_layer.size(); i++) {
            if ((constants[i-1]->rows() != nodes_per_layer[i]) || (constants[i-1]->cols() != 1))
                throw length_error("Input constants do not have the correct dimensions");

            if ((weights[i-1]->rows() != nodes_per_layer[i]) || (weights[i-1]->cols() != nodes_per_layer[i-1]))
                throw length_error("Input weights do not have the correct dimensions");                
        }
        this->constants = constants;
        this->weights = weights;
    }

    void NeuralNetwork::setup_topology(const vector<unsigned long> nodes_per_layer, const float learning_rate)
    {
        this->nodes_per_layer = nodes_per_layer;
        this->learning_rate = learning_rate;

        neuronLayers =  vector<shared_ptr<ColVector>>(nodes_per_layer.size());
        cacheLayers =   vector<shared_ptr<ColVector>>(nodes_per_layer.size());
        deltas =        vector<shared_ptr<ColVector>>(nodes_per_layer.size());

        // We may or may not be training here but these deltas are small so even if 
        // we are not training, the overhead is small.
        d_weights =       vector<shared_ptr<Matrix>>(nodes_per_layer.size() - 1);
        d_constants =     vector<shared_ptr<ColVector>>(nodes_per_layer.size() - 1);

        for (size_t i = 0; i < nodes_per_layer.size(); i++) {
            // initialize neuron layers
            neuronLayers[i] = make_shared<ColVector>(ColVector(nodes_per_layer[i]));

            // initialize cache and delta vectors
            cacheLayers[i] = make_shared<ColVector>(ColVector(nodes_per_layer[i]));
            deltas[i] =      make_shared<ColVector>(ColVector(nodes_per_layer[i]));

            // initialize weights and constants matrix
            if (i > 0) {
                d_constants[i-1] = make_shared<ColVector>(nodes_per_layer[i], 1);
                d_constants[i-1]->setZero();
                d_weights[i-1] = make_shared<Matrix>(nodes_per_layer[i], nodes_per_layer[i-1]);
                d_weights[i-1]->setZero();
            }
        }
        set_output_directory("E:/Code/kaggle/digits/data/");
    }

    void NeuralNetwork::set_output_directory(string output_directory)
    {
        this->output_directory = output_directory;
    }


    void  NeuralNetwork::propagate_forward(ColVector& input)
    {
        // set the input to input layer
        // block returns a part of the given vector or matrix
        // block takes 4 arguments : startRow, startCol, blockRows, blockCols
        neuronLayers.front()->block(0,0,neuronLayers.front()->size(),1) = input;
        // string filename = output_directory + "test_X.csv";
        // writeData(filename, *(neuronLayers.front()));

        // propagate the data forward and then
        // apply the activation function to your network
        // unaryExpr applies the given function to all elements of CURRENT_LAYER
        for (size_t i = 1; i < nodes_per_layer.size(); i++) 
        {
            (*cacheLayers[i]) = ((*weights[i - 1]) * (*neuronLayers[i - 1]) + (*constants[i-1]));
            // filename = output_directory + "test_cache_" + to_string(i) + ".csv";
            // writeData(filename, *cacheLayers[i]);
            (*neuronLayers[i]) = (*cacheLayers[i]).unaryExpr(&activation_function) ;
            // filename = output_directory + "test_neuron_" + to_string(i) + ".csv";
            // writeData(filename, *neuronLayers[i]);
        }
        // string filename = output_directory + "test_hatY.csv";
        // writeData(filename, *neuronLayers[nodes_per_layer.size()-1]);
    }

    void NeuralNetwork::propagate_backward(ColVector& output)
    {
        size_t final_layer_index = nodes_per_layer.size() - 1;
        // deltas = dA. 
        // Start with the final node where we need A = \mathscr{L} (\hat{y^[i]}, y^[i]) for i = 1,...,m
        for (size_t j = 0; j < nodes_per_layer[final_layer_index]; ++j)
        {
            (*deltas.back())(j) =  -(output(j) / (*neuronLayers[final_layer_index])(j) - 
                                    (1 - output(j)) / (1-(*neuronLayers[final_layer_index])(j)) ); 
        }
       
        // calculate the errors made by neurons of last layer
        for (int i = nodes_per_layer.size() - 2; i >= 0; --i)
        {
            // Ref: https://eigen.tuxfamily.org/dox/group__TutorialArrayClass.html: arrays interpret multiplication as coefficient-wise product
            // ColVector dz = (*cacheLayers[i+1]).unaryExpr(&activation_function_derivative); // dz_curr = dA_curr * sigmoid_backward(Z_curr)
            ColVector dz = (*deltas[i+1]).cwiseProduct((*cacheLayers[i+1]).unaryExpr(&activation_function_derivative)); // dz_curr = dA_curr * sigmoid_backward(Z_curr)
            // string filename = output_directory + "test_delta" + to_string(i) + ".csv";
            // writeData(filename, *deltas[i+1]);
            // filename = output_directory + "test_cache" + to_string(i) + ".csv";
            // writeData(filename, *cacheLayers[i+1]);

            (*deltas[i]) = (weights[i]->transpose()) * dz; // dA_prev = np.dot(W_curr.T, dZ_curr)                
            // filename = output_directory + "test_dA" + to_string(i) +".csv";
            // writeData(filename, (*deltas[i]));

            // Update Weights
            (*d_weights[i]) += (dz * (neuronLayers[i]->transpose()));
            // Matrix dw = dz * (neuronLayers[i]->transpose()); // dW_curr = np.dot(dZ_curr, A_prev.T) / m
            // filename = output_directory + "test_dW" + to_string(i) + ".csv";
            // writeData(filename, (*d_weights[i]));

            // Update constants
            // ColVector db = dz; // db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
            (*d_constants[i]) += dz;
            // string filename = output_directory + "test_b" + to_string(i) + ".csv";
            // writeData(filename, (*d_constants[i]));            
        }
    }

    void NeuralNetwork::write_parameters_to_file(string parameter_file_prefix)
    {
        for (size_t i = 0; i < weights.size(); ++ i)
        {
            string filename = output_directory + parameter_file_prefix + "W" + to_string(i) + ".csv";
            writeData(filename, (*weights[i]));
            filename = output_directory + parameter_file_prefix + "b" + to_string(i) + ".csv";
            writeData(filename, (*constants[i]));
        }
    }

    void NeuralNetwork::update_weights(size_t number_of_training_examples)
    {
        for (int i = nodes_per_layer.size() - 2; i >= 0; --i)
        {
            (*weights[i]) -= learning_rate / ((Scalar) number_of_training_examples) * (*d_weights[i]);

            (*constants[i]) -= learning_rate / ((Scalar) number_of_training_examples) * (*d_constants[i]);
        }
    }

    void NeuralNetwork::train(vector<shared_ptr<ColVector>> training_data, 
                             vector<shared_ptr<ColVector>> training_labels,
                             size_t epochs)
    {

        size_t m = training_data.size();
        Scalar cost = 0;
        Scalar accuracy = 0;
        int output_stats = 50;
        for (size_t i = 0; i < epochs; ++i)
        {
            cost = 0;
            accuracy = 0;
            for (size_t j = 0; j < m; ++j)
            {
                propagate_forward(*training_data[j]);
                if (i % output_stats == 0)
                {
                    cost += get_cost_value((neuronLayers.back()), training_labels[j]);
                    accuracy += (Scalar) get_accuracy((neuronLayers.back()), training_labels[j]);
                }
                propagate_backward(*training_labels[j]);
            }
            update_weights(m);
            // reset deltas
            for (size_t i = 0; i < d_constants.size(); ++i) 
            {
                d_constants[i]->setZero();
                d_weights[i]->setZero();
            }

            if (i % output_stats == 0)
            {
                cost /= m;
                accuracy /= m;
                cout << "Epochs: " << to_string(i) << ", cost: " << to_string(cost) << ", accuracy: " << to_string(accuracy) << endl;
                // write_parameters_to_file("tmp");
            }
        }

        write_parameters_to_file("tmp");
    }


 
    vector<unsigned long> NeuralNetwork::get_nodes_per_layer()
    {
        return nodes_per_layer;
    }

    shared_ptr<ColVector> NeuralNetwork::final_layer_after_forward_propagation()
    {
        return (neuronLayers.back());
    }
    */