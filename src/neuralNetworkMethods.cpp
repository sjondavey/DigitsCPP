#include "neuralNetworkMethods.h"

using namespace std;

namespace neuralnetworkfirstprinciples {

    void train_vectorized(const vector<unsigned short> nodes_per_layer, const float learning_rate, 
                          vector<shared_ptr<ColVector>>& constants, vector<shared_ptr<Matrix>>& weights,
                          shared_ptr<Matrix> data, shared_ptr<Matrix> labels,
                          size_t epochs,
                          size_t output_cost_accuracy_every_n_steps)
    {
        size_t last_layer = nodes_per_layer.size() - 1;
        vector<shared_ptr<Matrix>> A = vector<shared_ptr<Matrix>>(nodes_per_layer.size());
        vector<shared_ptr<Matrix>> dA = vector<shared_ptr<Matrix>>(nodes_per_layer.size());

        vector<shared_ptr<Matrix>> dW = vector<shared_ptr<Matrix>>(nodes_per_layer.size()-1);
        vector<shared_ptr<ColVector>> db = vector<shared_ptr<ColVector>>(nodes_per_layer.size()-1);
        vector<shared_ptr<Matrix>> Z = vector<shared_ptr<Matrix>>(nodes_per_layer.size()-1);
        vector<shared_ptr<Matrix>> dZ = vector<shared_ptr<Matrix>>(nodes_per_layer.size()-1);

        for (size_t i = 0; i < nodes_per_layer.size(); ++i)
        {
            A[i] = make_shared<Matrix>(nodes_per_layer[i], data->cols());
            dA[i] = make_shared<Matrix>(nodes_per_layer[i], data->cols());
            if (i > 0)
            {
                dW[i-1] = make_shared<Matrix>(weights[i-1]->rows(), weights[i-1]->cols());
                db[i-1] = make_shared<ColVector>(constants[i-1]->rows());
                Z[i-1] = make_shared<Matrix>(constants[i-1]->rows(), data->cols());
                dZ[i-1] = make_shared<Matrix>(constants[i-1]->rows(), data->cols());
            }
        }
        unique_ptr<Matrix> ones = make_unique<Matrix>(labels->rows(), labels->cols());
        ones->setOnes();
        for (size_t epoch = 0; epoch < epochs; ++epoch)
        {
            A[0] = data;
            size_t m = data->cols();
            // Forward Propagation
            for (size_t i = 0; i < last_layer; ++i)
            {
                *Z[i] = (*weights[i]) * (*A[i]);
                (*Z[i]).colwise() += (*constants[i]);             
                // string path = "E:/Code/kaggle/digits/test/";
                // string filename = path + "Z" + to_string(i) + ".csv";
                // write_data(filename, *Z[i]);
                (*A[i+1]) = (*Z[i]).unaryExpr(&activation_function);
                // filename = path + "A" + to_string(i+1) + ".csv";
                // write_data(filename, *A[i+1]);
            }
            // string filename = path + "A_" + to_string(last_layer) + ".csv";
            // write_matrix_to_file(filename, *A[last_layer]);

            // Back Propagation
            (*dA[last_layer]) = -((*labels).cwiseQuotient(*A[last_layer]) - ((*ones) - (*labels)).cwiseQuotient(((*ones) - (*A[last_layer]) )));
            // filename = path + "dA_" + to_string(last_layer) + ".csv";
            // write_matrix_to_file(filename, *dA[last_layer]);
            for (int i = last_layer-1; i >= 0; --i)
            {
                (*dZ[i]) = (*dA[i+1]).array() * ((*Z[i]).unaryExpr(&activation_function_derivative)).array();
                // string filename = path + "dZ" + to_string(i) + ".csv";
                // write_matrix_to_file(filename, *dZ[i]);
                (*dW[i]) = (*dZ[i]) * (*A[i]).transpose() / m;
                // filename = path + "dW" + to_string(i) + ".csv";
                // write_matrix_to_file(filename, *dW[i]);
                (*db[i]) = (*dZ[i]).rowwise().sum() / m;
                // filename = path + "db" + to_string(i) + ".csv";
                // write_matrix_to_file(filename, *db[i]);
                if (i > 0) // don't need to do this last calc. It makes a massive performance improvement
                    (*dA[i]) = (*weights[i]).transpose() * (*dZ[i]);
                    // filename = path + "dA" + to_string(i) + ".csv";
                    // write_matrix_to_file(filename, *dA[i]);
            }
            // Update parameters
            for (size_t i = 0; i < last_layer; ++i)
            {
                (*weights[i]) -= learning_rate * (*dW[i]);
                (*constants[i]) -= learning_rate * (*db[i]);
            }
            if (epoch % output_cost_accuracy_every_n_steps == 0)
            {
                Scalar cost = get_cost_value((A[last_layer]), labels);
                Scalar accuracy = get_accuracy((A[last_layer]), labels);
                cout << "Epochs: " << std::setfill('0') << std::setw(5) 
                     << to_string(epoch) << ", cost: " << to_string(cost) << ", accuracy: " << to_string(accuracy) << endl;
            }
        }
    }

    shared_ptr<Matrix> test_vectorized(vector<shared_ptr<ColVector>>& constants, vector<shared_ptr<Matrix>>& weights,
                                    shared_ptr<Matrix> data, shared_ptr<Matrix> labels)
    {
        vector<shared_ptr<Matrix>> A = vector<shared_ptr<Matrix>>(constants.size()+1);
        vector<shared_ptr<Matrix>> Z = vector<shared_ptr<Matrix>>(constants.size());

        A[0] = make_shared<Matrix>(data->rows(), data->cols());
        for (size_t i = 1; i < constants.size() + 1; ++i)
        {
            A[i] = make_shared<Matrix>(constants[i-1]->rows(), data->cols());
            Z[i-1] = make_shared<Matrix>(constants[i-1]->rows(), data->cols());
        }

        A[0] = data;
        size_t m = data->cols();
        // Forward Propagation
        for (size_t i = 0; i < constants.size(); ++i)
        {
            *Z[i] = (*weights[i]) * (*A[i]);
            (*Z[i]).colwise() += (*constants[i]);             
            (*A[i+1]) = (*Z[i]).unaryExpr(&activation_function);
        }
        return A[constants.size()];
    }


    void train_loop_faster(const vector<unsigned short> nodes_per_layer, const float learning_rate, 
                          vector<shared_ptr<ColVector>>& constants, vector<shared_ptr<Matrix>>& weights,
                          vector<shared_ptr<ColVector>>& data, vector<shared_ptr<ColVector>>& labels,
                          size_t epochs,
                          size_t output_cost_accuracy_every_n_steps)
    {
        Scalar cost = 0;
        Scalar accuracy = 0;

        double start; 
        double end; 
        size_t m = data.size();
        size_t final_layer_index = nodes_per_layer.size() - 1;
        unique_ptr<ColVector> ones = make_unique<ColVector>(nodes_per_layer[final_layer_index]);
        ones->setOnes();

        //////////////////// Setup private variables used by each threads to avoid race conditions ////////////////////
        // TODO : Why use shared_pointers here ?
        vector<shared_ptr<Matrix>> weights_transpose = vector<shared_ptr<Matrix>>(weights.size());
        for (size_t layer_count = 0; layer_count < weights.size(); ++layer_count)
        {
            (weights_transpose[layer_count]) = make_shared<Matrix>((weights[layer_count])->transpose());
        }

        vector<unique_ptr<Matrix>> d_weights_transpose      = vector<unique_ptr<Matrix>>(nodes_per_layer.size() - 1);
        vector<unique_ptr<ColVector>> d_constants = vector<unique_ptr<ColVector>>(nodes_per_layer.size() - 1);
        for (size_t nodes_count = 0; nodes_count < nodes_per_layer.size(); nodes_count++) {
            if (nodes_count > 0) {
                d_constants[nodes_count-1] = make_unique<ColVector>(nodes_per_layer[nodes_count], 1);
                d_weights_transpose[nodes_count-1]   = make_unique<Matrix>(nodes_per_layer[nodes_count-1], nodes_per_layer[nodes_count]);
            }
        }

        Scalar cost_private, accuracy_private;
        
        // Start the big loop
        for (int i = 0; i < epochs; ++i)
        {   
            cost = 0;
            accuracy = 0;         
            // reset all deltas to zero
            cost_private = 0;
            accuracy_private = 0;
            for (size_t nodes_count = 0; nodes_count < nodes_per_layer.size() - 1; nodes_count++) {
                d_constants[nodes_count]->setZero();
                d_weights_transpose[nodes_count]->setZero();
            }
            // iterate through each example in the training set
            Eigen::initParallel(); // https://eigen.tuxfamily.org/dox/TopicMultiThreading.html
            concurrency::parallel_for<size_t>(size_t(0), m, [&](size_t j)
            //for (size_t j = 0; j < m; ++j)
            {
                vector<unique_ptr<ColVector>> neuron_values =  vector<unique_ptr<ColVector>>(nodes_per_layer.size()); // A = sigmoid(Z)
                vector<unique_ptr<ColVector>> unactivated_values  =  vector<unique_ptr<ColVector>>(nodes_per_layer.size()); // Z
                vector<unique_ptr<ColVector>> d_neuron_values        =  vector<unique_ptr<ColVector>>(nodes_per_layer.size()); // dA
                // partial derivatives 
                vector<unique_ptr<ColVector>> d_unactivated_values         = vector<unique_ptr<ColVector>>(nodes_per_layer.size() - 1);
                for (size_t nodes_count = 0; nodes_count < nodes_per_layer.size(); nodes_count++) {
                    neuron_values[nodes_count]       = make_unique<ColVector>(ColVector(nodes_per_layer[nodes_count]));
                    unactivated_values[nodes_count]  = make_unique<ColVector>(ColVector(nodes_per_layer[nodes_count]));
                    d_neuron_values[nodes_count]     = make_unique<ColVector>(ColVector(nodes_per_layer[nodes_count]));
                    if (nodes_count > 0) 
                    {
                        d_unactivated_values[nodes_count-1]    = make_unique<ColVector>(nodes_per_layer[nodes_count], 1);
                    }
                }

                // forward propagation starts
                // step 1, set the input neurons to the specific training example
                (*neuron_values.front()) = (*data[j]);
                //neuron_layers.front()->block(0,0,neuron_layers.front()->size(),1) = (*data[j]);
                
                // Step 2, move the input through the layers, saving information that will be needed in the back propagation
                // apply the activation function to your network
                // unaryExpr applies the given function to all elements of CURRENT_LAYER
                for (size_t node_count = 1; node_count < nodes_per_layer.size(); node_count++) 
                {
                    (*unactivated_values[node_count]) = ((*weights[node_count - 1]) * (*neuron_values[node_count - 1]) + (*constants[node_count-1]));
                    (*neuron_values[node_count]) = (*unactivated_values[node_count]).unaryExpr(&activation_function) ;

                }                
                string path = "E:/Code/kaggle/digits/data/mess/";
                string filename = path + "neuron_values_" + to_string(j) + "_fast.csv";
                write_matrix_to_file(filename, *neuron_values.back());

                if (i % output_cost_accuracy_every_n_steps == 0) 
                { // Collect some stats
                    // cost_private += get_cost_value(neuron_values.back(), labels[j]);

                    cost_private -= 
                          ((*labels[j]).cwiseProduct((*neuron_values.back()).unaryExpr<Scalar(*)(Scalar)>(&std::log)) +
                          (*ones - *labels[j]).cwiseProduct((*ones - *neuron_values.back()).unaryExpr<Scalar(*)(Scalar)>(&std::log))).sum();

                    // Scalar sum = 0;
                    // for (size_t i = 0; i < (neuron_values.back())->size(); ++ i)
                    //     sum += (*labels[j])(i) * log((*(neuron_values.back()))(i)) + (1 - (*labels[j])(i)) * log(1-(*(neuron_values.back()))(i));
                    // cost_private -= sum;

                    // accuracy_private += get_accuracy(neuron_values.back(), labels[j]);
                    size_t i = 0;
                    bool match = true;
                    int match_as_int = 1;
                    while (match && i < (neuron_values.back())->size())
                    {
                        Scalar estimate = (*(neuron_values.back()))(i);
                        if (estimate > 0.5)
                            estimate = 1;
                        else
                            estimate = 0;
                        if (abs(estimate - (*labels[j])(i)) > 1e-3) // elements don't match
                        {
                            match = false;
                            match_as_int = 0;
                        }
                        ++i;
                    }
                    accuracy_private += (Scalar) match_as_int;
                }

                //////////////////////////////////////////////////////////////////////////////////
                // backwards propagation
                // Step 1: Start with the final node where where we need to calculate
                // A = \mathscr{L} (\hat{y^[i]}, y^[i]) for i = 1,...,m
                // one node at a time
                (*d_neuron_values.back()) = -((*labels[j]).cwiseQuotient(*neuron_values[final_layer_index]) - 
                                              ((*ones) - (*labels[j])).cwiseQuotient((*ones) - (*neuron_values[final_layer_index])) );


                // Step 2: move the error back in time to the first layer
                for (int k = nodes_per_layer.size() - 2; k >= 0; --k)
                {
                    (*d_unactivated_values[k]) = (*d_neuron_values[k+1]).cwiseProduct((*unactivated_values[k+1]).unaryExpr(&activation_function_derivative)); // dz_curr = dA_curr * sigmoid_backward(Z_curr)                            
                    // See http://eigen.tuxfamily.org/dox-devel/TopicWritingEfficientProductExpression.html for the use of .noalias()
                    (*d_weights_transpose[k]).noalias() += (*neuron_values[k]) * d_unactivated_values[k]->transpose();
                    (*d_constants[k]).noalias() += (*d_unactivated_values[k]);
                    if (k > 0)
                    {
                        (*d_neuron_values[k]) = (*weights_transpose[k]) * (*d_unactivated_values[k]); // dA_prev = np.dot(W_curr.T, dZ_curr)        
                    }
                }
            });
            for (int k = nodes_per_layer.size() - 2; k >= 0; --k)
            {
                (*weights_transpose[k]).noalias() -= learning_rate / ((Scalar) m) * (*d_weights_transpose[k]);
                (*weights[k]) = weights_transpose[k]->transpose();
                (*constants[k]).noalias() -= learning_rate / ((Scalar) m) * (*d_constants[k]);
            }

            if (i % output_cost_accuracy_every_n_steps == 0)
            {
                cost += cost_private;
                cost /= m;
                accuracy += accuracy_private;
                accuracy /= m;
                cout << "Epochs: " << std::setfill('0') << std::setw(5) 
                     << to_string(i) << ", cost: " << to_string(cost) << ", accuracy: " << to_string(accuracy) << endl;
            }
        } // for epochs

    } // one_method_train


/*
    void train_loop_faster(const vector<unsigned short> nodes_per_layer, const float learning_rate, 
                          vector<shared_ptr<ColVector>>& constants_input, vector<shared_ptr<Matrix>>& weights_input,
                          vector<shared_ptr<ColVector>>& data, vector<shared_ptr<ColVector>>& labels,
                          size_t epochs,
                          size_t output_cost_accuracy_every_n_steps)
    {
        Scalar cost = 0;
        Scalar accuracy = 0;


        double start; 
        double end; 
        size_t m = data.size();
        size_t final_layer_index = nodes_per_layer.size() - 1;
        unique_ptr<ColVector> ones = make_unique<ColVector>(nodes_per_layer[final_layer_index]);
        ones->setOnes();
        //////////////////// Setup private variables used by each threads to avoid race conditions ////////////////////
        // TODO : Why use shared_pointers here ?
        vector<shared_ptr<ColVector>> constants = constants_input;
        vector<shared_ptr<Matrix>> weights = weights_input;
        vector<shared_ptr<Matrix>> weights_transpose = vector<shared_ptr<Matrix>>(weights_input.size());
        for (size_t layer_count = 0; layer_count < weights_input.size(); ++layer_count)
        {
            (weights_transpose[layer_count]) = make_shared<Matrix>((weights[layer_count])->transpose());
        }

        vector<unique_ptr<ColVector>> neuron_values =  vector<unique_ptr<ColVector>>(nodes_per_layer.size()); // A = sigmoid(Z)
        vector<unique_ptr<ColVector>> unactivated_values  =  vector<unique_ptr<ColVector>>(nodes_per_layer.size()); // Z
        vector<unique_ptr<ColVector>> d_neuron_values        =  vector<unique_ptr<ColVector>>(nodes_per_layer.size()); // dA
        // partial derivatives 
        vector<unique_ptr<ColVector>> d_unactivated_values         = vector<unique_ptr<ColVector>>(nodes_per_layer.size() - 1);
        vector<unique_ptr<Matrix>> d_weights_transpose      = vector<unique_ptr<Matrix>>(nodes_per_layer.size() - 1);
        vector<unique_ptr<ColVector>> d_constants = vector<unique_ptr<ColVector>>(nodes_per_layer.size() - 1);
        for (size_t nodes_count = 0; nodes_count < nodes_per_layer.size(); nodes_count++) {
            neuron_values[nodes_count] = make_unique<ColVector>(ColVector(nodes_per_layer[nodes_count]));
            unactivated_values[nodes_count]  = make_unique<ColVector>(ColVector(nodes_per_layer[nodes_count]));
            d_neuron_values[nodes_count]        = make_unique<ColVector>(ColVector(nodes_per_layer[nodes_count]));

            if (nodes_count > 0) {
                d_unactivated_values[nodes_count-1]         = make_unique<ColVector>(nodes_per_layer[nodes_count], 1);
                d_constants[nodes_count-1] = make_unique<ColVector>(nodes_per_layer[nodes_count], 1);
                d_weights_transpose[nodes_count-1]   = make_unique<Matrix>(nodes_per_layer[nodes_count-1], nodes_per_layer[nodes_count]);
            }
        }

        Scalar cost_private, accuracy_private;
        
        // Start the big loop
        for (int i = 0; i < epochs; ++i)
        {   
            cost = 0;
            accuracy = 0;         
            // reset all deltas to zero
            cost_private = 0;
            accuracy_private = 0;
            for (size_t nodes_count = 0; nodes_count < nodes_per_layer.size() - 1; nodes_count++) {
                d_constants[nodes_count]->setZero();
                d_weights_transpose[nodes_count]->setZero();
            }
            // iterate through each example in the training set
            for (size_t j = 0; j < m; ++j)
            {
                // forward propagation starts
                // step 1, set the input neurons to the specific training example
                (*neuron_values.front()) = (*data[j]);
                //neuron_layers.front()->block(0,0,neuron_layers.front()->size(),1) = (*data[j]);
                
                // Step 2, move the input through the layers, saving information that will be needed in the back propagation
                // apply the activation function to your network
                // unaryExpr applies the given function to all elements of CURRENT_LAYER
                for (size_t node_count = 1; node_count < nodes_per_layer.size(); node_count++) 
                {
                    (*unactivated_values[node_count]) = ((*weights[node_count - 1]) * (*neuron_values[node_count - 1]) + (*constants[node_count-1]));
                    (*neuron_values[node_count]) = (*unactivated_values[node_count]).unaryExpr(&activation_function) ;
                }                
                if (i % output_cost_accuracy_every_n_steps == 0) 
                { // Collect some stats
                    // cost_private += get_cost_value(neuron_values.back(), labels[j]);

                    cost_private -= 
                          ((*labels[j]).cwiseProduct((*neuron_values.back()).unaryExpr<Scalar(*)(Scalar)>(&std::log)) +
                          (*ones - *labels[j]).cwiseProduct((*ones - *neuron_values.back()).unaryExpr<Scalar(*)(Scalar)>(&std::log))).sum();

                    // Scalar sum = 0;
                    // for (size_t i = 0; i < (neuron_values.back())->size(); ++ i)
                    //     sum += (*labels[j])(i) * log((*(neuron_values.back()))(i)) + (1 - (*labels[j])(i)) * log(1-(*(neuron_values.back()))(i));
                    // cost_private -= sum;

                    // accuracy_private += get_accuracy(neuron_values.back(), labels[j]);
                    size_t i = 0;
                    bool match = true;
                    int match_as_int = 1;
                    while (match && i < (neuron_values.back())->size())
                    {
                        Scalar estimate = (*(neuron_values.back()))(i);
                        if (estimate > 0.5)
                            estimate = 1;
                        else
                            estimate = 0;
                        if (abs(estimate - (*labels[j])(i)) > 1e-3) // elements don't match
                        {
                            match = false;
                            match_as_int = 0;
                        }
                        ++i;
                    }
                    accuracy_private += (Scalar) match_as_int;
                }

                //////////////////////////////////////////////////////////////////////////////////
                // backwards propagation
                // Step 1: Start with the final node where where we need to calculate
                // A = \mathscr{L} (\hat{y^[i]}, y^[i]) for i = 1,...,m
                // one node at a time
                (*d_neuron_values.back()) = -((*labels[j]).cwiseQuotient(*neuron_values[final_layer_index]) - 
                                              ((*ones) - (*labels[j])).cwiseQuotient((*ones) - (*neuron_values[final_layer_index])) );

                // Step 2: move the error back in time to the first layer
                for (int k = nodes_per_layer.size() - 2; k >= 0; --k)
                {
                    (*d_unactivated_values[k]) = (*d_neuron_values[k+1]).cwiseProduct((*unactivated_values[k+1]).unaryExpr(&activation_function_derivative)); // dz_curr = dA_curr * sigmoid_backward(Z_curr)                            
                    // See http://eigen.tuxfamily.org/dox-devel/TopicWritingEfficientProductExpression.html for the use of .noalias()
                    (*d_weights_transpose[k]).noalias() += (*neuron_values[k]) * d_unactivated_values[k]->transpose();
                    (*d_constants[k]).noalias() += (*d_unactivated_values[k]);
                    if (k > 0)
                    {
                        (*d_neuron_values[k]) = (*weights_transpose[k]) * (*d_unactivated_values[k]); // dA_prev = np.dot(W_curr.T, dZ_curr)        
                    }
                }
            }
            for (int k = nodes_per_layer.size() - 2; k >= 0; --k)
            {
                (*weights_transpose[k]).noalias() -= learning_rate / ((Scalar) m) * (*d_weights_transpose[k]);
                (*weights[k]) = weights_transpose[k]->transpose();
                (*constants[k]).noalias() -= learning_rate / ((Scalar) m) * (*d_constants[k]);
            }


            if (i % output_cost_accuracy_every_n_steps == 0)
            {
                cost += cost_private;
                cost /= m;
                accuracy += accuracy_private;
                accuracy /= m;
                cout << "Epochs: " << std::setfill('0') << std::setw(5) 
                     << to_string(i) << ", cost: " << to_string(cost) << ", accuracy: " << to_string(accuracy) << endl;
            }

        } // for epochs

    } // one_method_train
*/










    void train_loop_base(const vector<unsigned short> nodes_per_layer, const float learning_rate, 
                          vector<shared_ptr<ColVector>>& constants_input, vector<shared_ptr<Matrix>>& weights_input,
                          vector<shared_ptr<ColVector>>& data, vector<shared_ptr<ColVector>>& labels,
                          size_t epochs,
                          size_t output_cost_accuracy_every_n_steps)
    {
        Scalar cost = 0;
        Scalar accuracy = 0;

        //////////////////// Quick sanity check on vector dimensions //////////////////// 
        if (constants_input.size() != nodes_per_layer.size() - 1)
            throw length_error("Input constants do not have the correct number of column vectors");
        if (weights_input.size() != nodes_per_layer.size() - 1)
            throw length_error("Input weights do not have the correct number of matrices");
        for (size_t i = 1; i < nodes_per_layer.size(); i++) {
            if ((constants_input[i-1]->rows() != nodes_per_layer[i]) || (constants_input[i-1]->cols() != 1))
                throw length_error("Input constants do not have the correct dimensions");
            if ((weights_input[i-1]->rows() != nodes_per_layer[i]) || (weights_input[i-1]->cols() != nodes_per_layer[i-1]))
                throw length_error("Input weights do not have the correct dimensions");                
        }

        //////////////////// Quick sanity check on vector dimensions //////////////////// 
        size_t m = data.size();
        //////////////////// Setup private variables used by each threads to avoid race conditions ////////////////////
        // TODO : Why use shared_pointers here ?
        vector<shared_ptr<ColVector>> constants = constants_input;
        vector<shared_ptr<Matrix>> weights = weights_input;


        vector<shared_ptr<ColVector>> neuron_layers =  vector<shared_ptr<ColVector>>(nodes_per_layer.size()); // Z = sigmoid(A)
        vector<shared_ptr<ColVector>> cache_layers  =  vector<shared_ptr<ColVector>>(nodes_per_layer.size()); // A
        vector<shared_ptr<ColVector>> deltas        =  vector<shared_ptr<ColVector>>(nodes_per_layer.size()); // dA
        // partial derivatives 
        vector<shared_ptr<ColVector>> d_z            = vector<shared_ptr<ColVector>>(nodes_per_layer.size() - 1);
        vector<shared_ptr<Matrix>> d_weights      = vector<shared_ptr<Matrix>>(nodes_per_layer.size() - 1);
        vector<shared_ptr<ColVector>> d_constants = vector<shared_ptr<ColVector>>(nodes_per_layer.size() - 1);
        for (size_t nodes_count = 0; nodes_count < nodes_per_layer.size(); nodes_count++) {
            neuron_layers[nodes_count] = make_shared<ColVector>(ColVector(nodes_per_layer[nodes_count]));
            cache_layers[nodes_count]  = make_shared<ColVector>(ColVector(nodes_per_layer[nodes_count]));
            deltas[nodes_count]        = make_shared<ColVector>(ColVector(nodes_per_layer[nodes_count]));
            if (nodes_count > 0) {
                d_z[nodes_count-1]         = make_shared<ColVector>(nodes_per_layer[nodes_count], 1);
                d_constants[nodes_count-1] = make_shared<ColVector>(nodes_per_layer[nodes_count], 1);
                d_weights[nodes_count-1]   = make_shared<Matrix>(nodes_per_layer[nodes_count], nodes_per_layer[nodes_count-1]);
            }
        }

        Scalar cost_private, accuracy_private;
        size_t final_layer_index = nodes_per_layer.size() - 1;
        
        // Start the big loop
        for (int i = 0; i < epochs; ++i)
        {   
            cost = 0;
            accuracy = 0;         
            // reset all deltas to zero
            cost_private = 0;
            accuracy_private = 0;
            for (size_t nodes_count = 0; nodes_count < nodes_per_layer.size() - 1; nodes_count++) {
                d_constants[nodes_count]->setZero();
                d_weights[nodes_count]->setZero();
            }
            // iterate through each example in the training set
            for (size_t j = 0; j < m; ++j)
            {
                // forward propagation starts
                // step 1, set the input neurons to the specific training example
                neuron_layers.front()->block(0,0,neuron_layers.front()->size(),1) = (*data[j]);
                // Step 2, move the input through the layers, saving information that will be needed in the back propagation
                // apply the activation function to your network
                // unaryExpr applies the given function to all elements of CURRENT_LAYER
                for (size_t node_count = 1; node_count < nodes_per_layer.size(); node_count++) 
                {
                    (*cache_layers[node_count]) = ((*weights[node_count - 1]) * (*neuron_layers[node_count - 1]) + (*constants[node_count-1]));
                    // string path = "E:/Code/kaggle/digits/data/";
                    // string filename = path + "cache_layer" + to_string(j) + "_" + to_string(node_count) + ".csv";
                    // write_matrix_to_file(filename, *cache_layers[node_count]);

                    (*neuron_layers[node_count]) = (*cache_layers[node_count]).unaryExpr(&activation_function) ;
                }
                string path = "E:/Code/kaggle/digits/data/mess/";
                string filename = path + "neuron_values_" + to_string(j) + "_base.csv";
                write_matrix_to_file(filename, *neuron_layers.back());

                if (i % output_cost_accuracy_every_n_steps == 0)
                { // Collect some stats
                    cost_private += get_cost_value((neuron_layers.back()), labels[j]);
                    accuracy_private += (Scalar) get_accuracy((neuron_layers.back()), labels[j]);
                }

                //////////////////////////////////////////////////////////////////////////////////
                // backwards propagation
                // Step 1: Start with the final node where where we need to calcualte
                // A = \mathscr{L} (\hat{y^[i]}, y^[i]) for i = 1,...,m
                // one node at a time
                for (size_t node_number = 0; node_number < nodes_per_layer[final_layer_index]; ++node_number)
                {
                    (*deltas.back())(node_number) =  -((*labels[j])(node_number) / (*neuron_layers[final_layer_index])(node_number) - 
                                            (1 - (*labels[j])(node_number)) / (1-(*neuron_layers[final_layer_index])(node_number)) ); 
                }
                // string path = "E:/Code/kaggle/digits/data/mess/";
                // string filename = path + "d_neuron_values_" + to_string(j) + "_base.csv";
                // write_matrix_to_file(filename, *deltas.back());

                // Step 2: move the error back in time to the first layer
                for (int k = nodes_per_layer.size() - 2; k >= 0; --k)
                {
                    (*d_z[k]) = (*deltas[k+1]).cwiseProduct((*cache_layers[k+1]).unaryExpr(&activation_function_derivative)); // dz_curr = dA_curr * sigmoid_backward(Z_curr)
                    // Update Weights
                    (*d_weights[k]) += ((*d_z[k]) * (neuron_layers[k]->transpose()));
                    (*d_constants[k]) += (*d_z[k]);
                    if (k > 0)
                    {
                        (*deltas[k]) = (weights[k]->transpose()) * (*d_z[k]); // dA_prev = np.dot(W_curr.T, dZ_curr)                
                    }

                }
            }
            for (int k = nodes_per_layer.size() - 2; k >= 0; --k)
            {
                (*weights[k]) -= learning_rate / ((Scalar) m) * (*d_weights[k]);
                (*constants[k]) -= learning_rate / ((Scalar) m) * (*d_constants[k]);
            }

            if (i % output_cost_accuracy_every_n_steps == 0)
            {
                cost += cost_private;
                cost /= m;
                accuracy += accuracy_private;
                accuracy /= m;
                cout << "Epochs: " << std::setfill('0') << std::setw(5) 
                     << to_string(i) << ", cost: " << to_string(cost) << ", accuracy: " << to_string(accuracy) << endl;
            }
        } // main if loop

    } // one_method_train

} // namespace neuralnetworkfirstprinciples