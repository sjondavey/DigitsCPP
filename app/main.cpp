#include <iostream>
#include "INIReader.h"
#include <string.h>
#include <sstream>
#include <vector>
#include "../src/neuralnetwork.h"
#include "../src/neuralNetworkMethods.h"
#include "../src/inputdatafilereader.h"
#include <omp.h>

using namespace std;
using namespace neuralnetworkfirstprinciples;


int run_nn()
{
    // There are vectorized and 'loop' versions of gradient descent. There take data and label inputs in 
    // different formats. Setting this value to true creates variables for vectorised versions and 
    // false for loop versions of the gradient descent function
    bool vectorised = true;

    std::cout << "Loading inputs";
    if (vectorised) 
    {
        cout << " for a vectorized gradient descent solution" << std::endl;
    }
    else {
        cout << " for a 'loop' gradient descent solution" << std::endl;
    }

    INIReader reader("../../examples/test.ini");

    if (reader.ParseError() < 0) {
        std::cout << "Can't load 'test.ini'\n";
        return 1;
    }

    ///////////////// NETWORK Parameters /////////////////
    // Read network::architecture and convert it into a vector of int
    vector<unsigned short> nodes_per_layer;
    string default_value = "Not Found";
    string architecture_as_string = reader.GetString("network", "architecture", default_value);
    if (architecture_as_string.compare(default_value) == 0)
    {
        cout << "Unable to read Section network::architecture. This should be a common delimited list of neurons per layer - including input and output layers" << endl;
        return 1;
    }
    stringstream s_stream(architecture_as_string); //create string stream from the string
    while(s_stream.good()) {
        string substr;
        getline(s_stream, substr, ','); //get first string delimited by comma
        try 
        {
            int nodes = stoi(substr);
            // TODO check int against 'short' boundary and throw range_error
            if (nodes < 1)
            {
                std::cout << "Nodes per layer must be a non-negative integer" << '\n';
                return -1;
            }
            cout << "Nodes per payer: " << substr << endl;
            nodes_per_layer.push_back(nodes);
        }
        catch (invalid_argument const& ex)
        {
            std::cout << "Unable to convert architecture into an integer: std::invalid_argument::what(): " << ex.what() << '\n';
        }
        catch(std::out_of_range const& ex)
        {
            std::cout << "Unable to convert architecture into an integer: std::out_of_range::what(): " << ex.what() << '\n';
        }        
    }
    std::cout << "NN Architecture: " << architecture_as_string << " specified" << std::endl;

    // Read network::learning_rate 
    float learning_rate = reader.GetReal("network", "learning_rate", -1);
    if (learning_rate < 0)
    {
        std::cout << "Learning rate must be a non-negative double" << '\n';
        return -1;        
    }
    std::cout << "NN learning rate: " << to_string(learning_rate) << std::endl;

    // Create the neural network with the input parameters
    
    vector<shared_ptr<ColVector>> constants;
    vector<shared_ptr<Matrix>> weights;
    bool use_existing_parameters = reader.GetBoolean("network", "use_existing_parameters", false);
    if (use_existing_parameters)
    {
        string existing_parameters_path = reader.GetString("network", "existing_parameters_path", "");
        if (existing_parameters_path == "")
        {
            cout << "Unable to read existing_parameters_path" << endl;
            return -1;
        }
        string existing_parameters_file_prefix = reader.GetString("network", "existing_parameters_file_prefix", "");
        if (existing_parameters_file_prefix == "")
        {
            cout << "Unable to read existing_parameters_file_prefix" << endl;
            return -1;
        }
        read_parameters(weights, constants, existing_parameters_path, existing_parameters_file_prefix);

        if (weights.empty())
        {
            cout << "Unable to read the input parameters" << endl;
            return -1;
        }
    }
    else
    {
        generate_random_weights(nodes_per_layer, weights, constants);
    }
    NeuralNetwork nn = NeuralNetwork(nodes_per_layer, learning_rate, constants, weights);

    ///////////////// DATA Parameters /////////////////
    // Read data::input_data_file 
    bool train = reader.GetBoolean("data", "train", false);
    if (train)
    {
        // Epochs
        int epochs = reader.GetInteger("data", "epochs", 0);
        if (epochs < 1)
        {
            std::cout << "Epochs must be a positive integer" << '\n';
            return -1;        
        }
        cout << "Training the nn with " << to_string(epochs) << " epochs" << endl;
        bool output_training_cost = reader.GetBoolean("data", "output_training_cost", false);
        int output_cost_every_n_epochs;
        if (!output_training_cost)
            output_cost_every_n_epochs = epochs + 1;
        else
            output_cost_every_n_epochs = reader.GetInteger("data", "output_cost_every_n_epochs", 10000);

        // Raw Data
        string input_data_file = reader.GetString("data", "input_data_file", default_value);
        if (input_data_file.compare(default_value) == 0)
        {
            cout << "Unable to read Section data::input_data_file. This should be a string file name" << endl;
            return 1;
        }
        std::cout << "Reading raw data from file: " << input_data_file << std::endl;

        std::ifstream file(input_data_file);
        if(file.fail()){
            cout << "Unable to find file " << input_data_file << endl;
            return -1;
        }

        shared_ptr<Matrix> labels; // for vectorized gradient descent
        shared_ptr<Matrix> data;   // for vectorized gradient descent
        vector<shared_ptr<ColVector>> labels_loop; // for gradient descent using loops
        vector<shared_ptr<ColVector>> data_loop;   // for gradient descent using loops
        size_t raw_data_volume, sample_data_length, sample_label_length;
        if (vectorised)
        {
            read_digit_data(input_data_file, labels, data);
            raw_data_volume = data->cols();
            sample_data_length = data->rows();
            sample_label_length = labels->rows();
        }
        else 
        {
            read_digit_data(input_data_file, labels_loop, data_loop);
            raw_data_volume = labels_loop.size();
            sample_data_length = (*data_loop[0]).rows();
            sample_label_length = labels_loop[0]->rows();
        }

        if(sample_data_length != nodes_per_layer[0])
        {
            cout << "Input architecture has " << to_string(nodes_per_layer[0]) << " features but input data has "
                 << to_string(sample_data_length) << " features" << endl; 
                 return -1;
        }
        if(sample_label_length != nodes_per_layer[nodes_per_layer.size()-1])
        {
            cout << "Input architecture has " << to_string(nodes_per_layer[nodes_per_layer.size() - 1]) << " features but input labels have "
                 << to_string(sample_label_length) << " features" << endl; 
                 return -1;
        }

        double training_data_split = reader.GetReal("data", "training_data_split", -1);    
        if ((training_data_split < 0) || (training_data_split > 1))
        {
            cout << "training_data_split must be in the range [0.0, 1.0]" << endl;
            return -1;
        }
        
        size_t training_data_volume = (size_t) (raw_data_volume * training_data_split);
        std::cout << "Total data points: " << to_string(raw_data_volume) << ", of which " 
            << to_string(training_data_volume) << " allocated to training and " 
            << to_string(raw_data_volume-training_data_volume) << " to testing"  << endl;

        shared_ptr<Matrix> training_data, training_labels, test_data, test_labels;
        vector<shared_ptr<ColVector>> training_data_loop, training_labels_loop, test_data_loop, test_labels_loop;
        if (vectorised)
        {
            split(data, labels, 
                training_data, training_labels, 
                test_data, test_labels,
                training_data_volume);   
            cout << "Total data points: " << to_string(raw_data_volume) << ", of which " 
                << to_string(training_data->cols()) << " allocated to training and " 
                << to_string(test_data->cols()) << " to testing"  << endl;
        }
        else
        {
            split(data_loop, labels_loop, 
                    training_data_loop, training_labels_loop, 
                    test_data_loop, test_labels_loop,
                    training_data_volume);   
            cout << "Total data points: " << to_string(raw_data_volume) << ", of which " 
                << to_string(training_data_loop.size()) << " allocated to training and " 
                << to_string(test_data_loop.size()) << " to testing"  << endl;
        }



        int processors = omp_get_num_procs();
        Eigen::setNbThreads(processors);
        cout << "Eigen number of threads: " << Eigen::nbThreads( ) << endl;

        double start = omp_get_wtime(); 

        if (vectorised)
        {
            cout << "Using Vector Class" << endl;
            nn.train(training_data, training_labels, epochs, output_cost_every_n_epochs);

            // cout << "Using Vector Method" << endl;
            // train_vectorized(nodes_per_layer, learning_rate, 
            //                 constants, weights,
            //                 training_data, training_labels,
            //                 epochs);
        }
        else
        {
            cout << "Using loops" << endl;
            one_method_train(nodes_per_layer, learning_rate, 
                            constants, weights,
                            training_data_loop, training_labels_loop,
                            epochs);
        }
        double end = omp_get_wtime(); 
        printf("Training took %f seconds\n", end - start);

        bool write_trained_parameters_to_file = reader.GetBoolean("data", "write_trained_parameters_to_file", false);
        if (write_trained_parameters_to_file)
        {
            string path_to_write_trained_parameters = reader.GetString("data", "path_to_write_trained_parameters", "./");
            string parameter_file_prefix = reader.GetString("data", "parameter_file_prefix", "default");
            write_parameters(weights, constants, path_to_write_trained_parameters, parameter_file_prefix);
            cout << "Training parameters persisted to file" << endl;
        }
        else {
            cout << "Training parameters NOT persisted to file" << endl;
        }
        ///////////////// TEST using training parameters /////////////////
        bool test_after_training = reader.GetBoolean("runtest", "test_after_training", false);
        if (test_after_training)
        {
            shared_ptr<Matrix> test_results;
            test_results = test_vectorized(constants, weights, data, labels);           
            cout << "Accuracy on test set: " << to_string(get_accuracy(test_results, labels)) << endl;
        }

    } // if(train)
    else // i.e. train == false; so use parameters from disk
    {
        bool test_after_training = reader.GetBoolean("runtest", "test_after_training", false);

        ///////////////// TEST using parameters on disk /////////////////
        bool test_data_from_file = reader.GetBoolean("runtest", "test_data_from_file", false);
        if (test_data_from_file)
        {
            string test_data_file = reader.GetString("runtest", "test_data_file", default_value);
            if (test_data_file.compare(default_value) == 0)
            {
                cout << "Unable to read Section test::test_data_file. This should be a string file name" << endl;
                return 1;
            }
            std::cout << "Reading raw data from file: " << test_data_file << std::endl;

            std::ifstream file(test_data_file);
            if(file.fail()){
                cout << "Unable to find file " << test_data_file << endl;
                return -1;
            }
            // Raw Data as Matrix
            shared_ptr<Matrix> labels;
            shared_ptr<Matrix> data;
            read_digit_data(test_data_file, labels, data);
            
            shared_ptr<Matrix> predicted_values;
            predicted_values = test_vectorized(constants, weights, data, labels);
            write_matrix_to_file("E:/Code/kaggle/digits/data/Y_Hat.csv", *predicted_values);
            write_matrix_to_file("E:/Code/kaggle/digits/data/Y.csv", *labels);
            
            cout << "Accuracy: " << to_string(get_accuracy(predicted_values, labels)) << endl;




        } // if (test_data_from_file)
     } // else part of if(train)
    return 0;
}



int main()
{
    return run_nn();    

    // vector<shared_ptr<ColVector>> global_constants(1); // size of vector will be > 1 in real life with size and length of ColVector know at runtime
    // global_constants[0] = make_shared<ColVector>(785);
    // global_constants[0]->setRandom();

    // size_t training_set_size = 10000;

    // double start; 
    // double end; 
    // start = omp_get_wtime(); 

    // for (size_t epochs = 0; epochs < 1000; ++epochs)
    // {
    //     vector<shared_ptr<ColVector>> changes_to_constants(global_constants.size()); 
    //     changes_to_constants[0] = make_shared<ColVector>(global_constants[0]->rows());
    //     changes_to_constants[0]->setZero();

    // //#pragma omp parallel for reduction(+:changes_to_constants)
    //     for (size_t example_number = 0; example_number < training_set_size; ++example_number)
    //     {
    //         //forward propagation
    //         ColVector some_random_result = (*global_constants[0]).cwiseProduct((*global_constants[0])); 
    //         //back propagation
    //         ColVector another_random_result = some_random_result.cwiseQuotient((*global_constants[0]));
    //         //update local constants
    //         (*changes_to_constants[0]) += 0.0001 * another_random_result;
    //     }
    //     // update global constants
    //     (*global_constants[0]) += (*changes_to_constants[0]);

    // }
    // end = omp_get_wtime(); 
    // printf("Work took %f seconds\n", end - start);
}
