#include "neuralnetworkloop.h"

using namespace std;

namespace neuralnetworkfirstprinciples {

    NeuralNetworkLoop::NeuralNetworkLoop(const vector<unsigned short> nodes_per_layer, 
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

        weights_transpose = vector<shared_ptr<Matrix>>(weights.size());
        for (size_t layer_count = 0; layer_count < weights.size(); ++layer_count)
        {
            (weights_transpose[layer_count]) = make_shared<Matrix>((weights[layer_count])->transpose());
        }
        this->training_epoch_number = 0;
    }

    NeuralNetworkLoop::NeuralNetworkLoop( NeuralNetworkLoop& nnl, tbb::split )
    {
        this->nodes_per_layer = nnl.nodes_per_layer;
        this->learning_rate = nnl.learning_rate;
        this->constants = nnl.constants;
        this->weights = nnl.weights;
        this->weights_transpose = nnl.weights_transpose;

        d_weights_transpose =       vector<shared_ptr<Matrix>>(nodes_per_layer.size() - 1);
        d_constants =     vector<shared_ptr<ColVector>>(nodes_per_layer.size() - 1);
        for (size_t i = 0; i < nodes_per_layer.size(); i++) {
            if (i > 0) {
                d_constants[i-1] = make_shared<ColVector>(nodes_per_layer[i]);
                d_constants[i-1]->setZero();
                d_weights_transpose[i-1]   = make_unique<Matrix>(nodes_per_layer[i-1], nodes_per_layer[i]);
                d_weights_transpose[i-1]->setZero();
            }
        }

        set_training_parameters(nnl.data, nnl.labels, nnl.epochs, nnl.output_cost_accuracy_every_n_steps);
        this->training_epoch_number = nnl.training_epoch_number;
    }

    void NeuralNetworkLoop::operator()(const tbb::blocked_range<size_t>& r)
    {
        vector<size_t> training_set_indicies(r.size());
        size_t counter = 0;
        for (auto it = r.begin(); it != r.end(); ++it)
        {
            training_set_indicies[counter] = (it);
            ++counter;
        }
        propagate(training_set_indicies);
    }

    void NeuralNetworkLoop::join(NeuralNetworkLoop& rhs)
    {
        cost += rhs.cost;
        accuracy += rhs.accuracy;
        for (int k = nodes_per_layer.size() - 2; k >= 0; --k)
        {
            (*d_weights_transpose[k]).noalias() += *rhs.d_weights_transpose[k];
            (*d_constants[k]).noalias() += *rhs.d_constants[k];
        }        
    }

    void NeuralNetworkLoop::set_input_data(vector<shared_ptr<ColVector>>& data)
    {
        this->data = data;
        number_of_training_examples = data.size();

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
        d_weights_transpose =       vector<shared_ptr<Matrix>>(nodes_per_layer.size() - 1);
        d_constants =     vector<shared_ptr<ColVector>>(nodes_per_layer.size() - 1);
        for (size_t i = 0; i < nodes_per_layer.size(); i++) {
            if (i > 0) {
                d_constants[i-1] = make_shared<ColVector>(nodes_per_layer[i]);
                d_constants[i-1]->setZero();
                d_weights_transpose[i-1]   = make_unique<Matrix>(nodes_per_layer[i-1], nodes_per_layer[i]);
                d_weights_transpose[i-1]->setZero();
            }
        }
        cost = 0;
        accuracy = 0;
    }

    void NeuralNetworkLoop::set_expected_results(vector<shared_ptr<ColVector>>& labels)
    {
        this->labels = labels;
        ones = make_unique<ColVector>(labels[0]->rows(), 1);
        ones->setOnes();
    }

    void NeuralNetworkLoop::propagate(vector<size_t> training_set_indicies)
    {
        // neuron_values[0] will be set equal to the input data so other vectors will be 'shorter' by 1 than this
        vector<unique_ptr<ColVector>> neuron_values        =  vector<unique_ptr<ColVector>>(nodes_per_layer.size()); // stores the different layers of out network (AKA 'activated' vector A = sigmoid(Z))
        vector<unique_ptr<ColVector>> unactivated_values   =  vector<unique_ptr<ColVector>>(nodes_per_layer.size() - 1); // stores the un-activated (activation fn not yet applied) values of layers (i.e. Z = WX + b)
        // partial derivatives 
        vector<unique_ptr<ColVector>> d_neuron_values      =  vector<unique_ptr<ColVector>>(nodes_per_layer.size() - 1); // stores the error contribution of each neurons (i.e. dA where A = sigmoid(Z))
        vector<unique_ptr<ColVector>> d_unactivated_values = vector<unique_ptr<ColVector>>(nodes_per_layer.size() - 1); // derivative of unactivated_values
        for (size_t nodes_count = 0; nodes_count < nodes_per_layer.size(); nodes_count++) 
        {
            neuron_values[nodes_count]              = make_unique<ColVector>(ColVector(nodes_per_layer[nodes_count]));
            if (nodes_count > 0) 
            {
                unactivated_values[nodes_count-1]   = make_unique<ColVector>(ColVector(nodes_per_layer[nodes_count]));
                d_neuron_values[nodes_count-1]      = make_unique<ColVector>(ColVector(nodes_per_layer[nodes_count]));
                d_unactivated_values[nodes_count-1] = make_unique<ColVector>(ColVector(nodes_per_layer[nodes_count]));
            }
        }
        bool collect_summary_stats = false;
        if (training_epoch_number % output_cost_accuracy_every_n_steps == 0)
            collect_summary_stats = true;
        for(std::vector<size_t>::iterator it = begin(training_set_indicies); it!=end(training_set_indicies); ++it) 
        {
            size_t j = *it;
            // forward propagation starts
            // step 1, set the input neurons to the specific training example
            (*neuron_values.front()) = (*data[j]);                
            // Step 2, move the input through the layers, saving information that will be needed in the back propagation
            // apply the activation function to your network
            // unaryExpr applies the given function to all elements of CURRENT_LAYER
            for (size_t node_count = 1; node_count < nodes_per_layer.size(); node_count++) 
            {
                (*unactivated_values[node_count-1]) = ((*weights[node_count - 1]) * (*neuron_values[node_count - 1]) + (*constants[node_count-1]));
                (*neuron_values[node_count]) = (*unactivated_values[node_count-1]).unaryExpr(&activation_function) ;
            }                
            if (collect_summary_stats)
            {
                    Scalar tmp = get_cost_value(neuron_values.back(), labels[j]);
                    cost += tmp;
                    // cost_private -= 
                    //       ((*labels[j]).cwiseProduct((*neuron_values.back()).unaryExpr<Scalar(*)(Scalar)>(&std::log)) +
                    //       (*ones - *labels[j]).cwiseProduct((*ones - *neuron_values.back()).unaryExpr<Scalar(*)(Scalar)>(&std::log))).sum();

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
                    accuracy += (Scalar) match_as_int;
            }

            // backwards propagation
            // Step 1: Start with the final node where where we need to calculate
            // A = \mathscr{L} (\hat{y^[i]}, y^[i]) for i = 1,...,m
            // one node at a time
            (*d_neuron_values.back()) = -(  (*labels[j]).cwiseQuotient(*neuron_values[nodes_per_layer.size() - 1]) - 
                                            ((*ones) - (*labels[j])).cwiseQuotient((*ones) - (*neuron_values[nodes_per_layer.size() - 1])) );

            // string path = "E:/Code/kaggle/digits/data/mess/";
            // string filename = path + "d_neuron_values_" + to_string(j) + "_class.csv";
            // write_matrix_to_file(filename, *d_neuron_values.back());

            // Step 2: move the error back in time to the first layer
            for (int k = nodes_per_layer.size() - 2; k >= 0; --k)
            {
                (*d_unactivated_values[k]) = (*d_neuron_values[k]).cwiseProduct((*unactivated_values[k]).unaryExpr(&activation_function_derivative)); // dz_curr = dA_curr * sigmoid_backward(Z_curr)                            
                // See http://eigen.tuxfamily.org/dox-devel/TopicWritingEfficientProductExpression.html for the use of .noalias()
                (*d_weights_transpose[k]).noalias() += (*neuron_values[k]) * d_unactivated_values[k]->transpose();
                (*d_constants[k]).noalias() += (*d_unactivated_values[k]);
                if (k > 0)
                {
                    (*d_neuron_values[k-1]) = (*weights_transpose[k]) * (*d_unactivated_values[k]); // dA_prev = np.dot(W_curr.T, dZ_curr)        
                }
            }
        }

    }

    void NeuralNetworkLoop::update_weights(size_t size_of_training_set_indicies)
    {
        bool collect_summary_stats = false;
        if (training_epoch_number % output_cost_accuracy_every_n_steps == 0)
            collect_summary_stats = true;

        size_t m = size_of_training_set_indicies;
        if (collect_summary_stats)
        {
            cost /= m;
            accuracy /= m;
            cout << "Epochs: " << std::setfill('0') << std::setw(5) << to_string(training_epoch_number);
            cout << ", cost: " << to_string(cost) << ", accuracy: " << to_string(accuracy) << endl;
        }

        for (int k = nodes_per_layer.size() - 2; k >= 0; --k)
        {
            (*weights_transpose[k]).noalias() -= learning_rate / ((Scalar) m) * (*d_weights_transpose[k]);
            d_weights_transpose[k]->setZero();
            (*weights[k]) = weights_transpose[k]->transpose();
            (*constants[k]).noalias() -= learning_rate / ((Scalar) m) * (*d_constants[k]);
            d_constants[k]->setZero();
        }
        training_epoch_number++;
    }

        
    shared_ptr<Matrix> NeuralNetworkLoop::evaluate(vector<shared_ptr<ColVector>>& data)
    {
        throw logic_error("Not yet implemented");
    }

    void NeuralNetworkLoop::set_training_parameters(vector<shared_ptr<ColVector>>& data,
                                                    vector<shared_ptr<ColVector>>& labels,
                                                    size_t epochs,
                                                    size_t output_cost_accuracy_every_n_steps)
    {
        set_input_data(data);
        set_expected_results(labels);
        this->epochs = epochs;
        this->output_cost_accuracy_every_n_steps = output_cost_accuracy_every_n_steps;
        cost = 0;
        accuracy = 0;
    }

    void NeuralNetworkLoop::train(vector<shared_ptr<ColVector>>& data,
               vector<shared_ptr<ColVector>>& labels,
               size_t epochs,
               size_t output_cost_accuracy_every_n_steps)
    {
        set_training_parameters(data, labels, epochs, output_cost_accuracy_every_n_steps);

        vector<size_t> all_indicies = vector<size_t>(data.size());
        std::iota(all_indicies.begin(), all_indicies.end(), 0);

        for (size_t i = 0; i < epochs; ++i)
        {
            bool output_summary_stats = false;
            propagate(all_indicies);
            update_weights(all_indicies.size());
        }
    }

    void NeuralNetworkLoop::train_parallel()
    {
        throw logic_error("Not yet implemented");

    }

 
    vector<unsigned short> NeuralNetworkLoop::get_nodes_per_layer()
    {
        return nodes_per_layer;
    }

}

