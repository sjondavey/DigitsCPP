#include "functions.h"

using namespace std;

namespace neuralnetworkfirstprinciples {

    ////////////////////////////// Just split into the first 'n' and the remaining "m-n'"    //////////////////////////////
    void split(vector<shared_ptr<ColVector>> full_data, 
            vector<shared_ptr<ColVector>> full_labels, 
            vector<shared_ptr<ColVector>>& training_data, 
            vector<shared_ptr<ColVector>>& training_labels, 
            vector<shared_ptr<ColVector>>& test_data, 
            vector<shared_ptr<ColVector>>& test_labels,
            size_t training_set_size)
    {
            training_data = vector<shared_ptr<ColVector>>(training_set_size);
            training_labels = vector<shared_ptr<ColVector>>(training_set_size);
            for (size_t i = 0; i < training_set_size; ++i)
            {
                training_data[i] = full_data[i];
                training_labels[i] = full_labels[i];
            }
            size_t len = full_data.size() - training_set_size;
            test_data = vector<shared_ptr<ColVector>>(len);
            test_labels = vector<shared_ptr<ColVector>>(len);
            size_t index = 0;
            for (size_t i = training_set_size; i < full_data.size(); ++i)
            {
                test_data[index] = full_data[i];
                test_labels[index] = full_labels[i];
                ++index;
            }
            return;
    }

    void split(shared_ptr<Matrix> full_data, 
           shared_ptr<Matrix> full_labels, 
           shared_ptr<Matrix>& training_data, 
           shared_ptr<Matrix>& training_labels, 
           shared_ptr<Matrix>& test_data, 
           shared_ptr<Matrix>& test_labels,
           size_t training_set_size)
    {
        training_data = make_shared<Matrix>(full_data->rows(), training_set_size);
        (*training_data) = full_data->block(0, 0, full_data->rows(), training_set_size);

        training_labels = make_shared<Matrix>(full_labels->rows(), training_set_size);
        (*training_labels) = full_labels->block(0, 0, full_labels->rows(), training_set_size);
        
        size_t test_set_size = full_data->cols() - training_set_size;
        test_data = make_shared<Matrix>(full_data->rows(), test_set_size);
        (*test_data) = full_data->block(0, training_set_size, full_data->rows(), test_set_size);

        test_labels = make_shared<Matrix>(full_labels->rows(), test_set_size);
        (*test_labels) = full_labels->block(0, training_set_size, full_labels->rows(), test_set_size);
    }


    ////////////////////////////// Activation    //////////////////////////////
    Scalar activation_function(Scalar x)
    {
        return 1 / (1 + exp(-x));
    }

    Scalar activation_function_derivative(Scalar x)
    {
        Scalar sig = 1 / (1 + exp(-x));
        return sig * (1 - sig);
    }

    Scalar get_cost_value(shared_ptr<ColVector> y_hat, shared_ptr<ColVector> y)
    {
        Scalar sum = 0;
        for (size_t i = 0; i < y_hat->size(); ++ i)
            sum += (*y)(i) * log((*y_hat)(i)) + (1 - (*y)(i)) * log(1-(*y_hat)(i));
        return -sum;
    }

    Scalar get_cost_value(unique_ptr<ColVector>& y_hat, shared_ptr<ColVector> y)
    {
        Scalar sum = 0;
        for (size_t i = 0; i < y_hat->size(); ++ i)
            sum += (*y)(i) * log((*y_hat)(i)) + (1 - (*y)(i)) * log(1-(*y_hat)(i));
        return -sum;
    }

    Scalar get_cost_value(shared_ptr<Matrix> Y_hat, shared_ptr<Matrix> Y)
    {
        Scalar sum = 0;
        for (size_t c = 0; c < Y_hat->cols(); ++ c)
        {
            for (size_t r = 0; r < Y_hat->rows(); ++r)
            {
                sum += (*Y)(r,c) * log((*Y_hat)(r,c)) + (1 - (*Y)(r,c)) * log(1-(*Y_hat)(r,c));
            }
        }
        return -sum /  Y_hat->cols();
    }

    int get_accuracy(shared_ptr<ColVector> y_hat, shared_ptr<ColVector> y)
    {
        for (size_t i = 0; i < y_hat->size(); ++ i)
        {
            Scalar estimate = (*y_hat)(i);
            if (estimate > 0.5)
                estimate = 1;
            else
                estimate = 0;
            if (abs(estimate - (*y)(i)) > 1e-3) // elements don't match
                return 0;
        }
        return 1; // all elements match
    }

    int get_accuracy(unique_ptr<ColVector>& y_hat, shared_ptr<ColVector> y)
    {
         for (size_t i = 0; i < y_hat->size(); ++ i)
        {
            Scalar estimate = (*y_hat)(i);
            if (estimate > 0.5)
                estimate = 1;
            else
                estimate = 0;
            if (abs(estimate - (*y)(i)) > 1e-3) // elements don't match
                return 0;
        }
        return 1; // all elements match
    }

    Scalar get_accuracy(shared_ptr<Matrix> Y_hat, shared_ptr<Matrix> Y)
    {
        Scalar sum = 0;
        for (size_t c = 0; c < Y_hat->cols(); ++ c)
        {
            size_t r = 0;
            bool match = true;
            int match_as_int = 1;
            while (match && r < (Y_hat->rows()))
            {
                Scalar estimate = (*Y_hat)(r,c);
                if (estimate > 0.5)
                    estimate = 1;
                else
                    estimate = 0;
                if (abs(estimate - (*Y)(r,c)) > 1e-3) // elements don't match
                {
                    match = false;
                    match_as_int = 0;
                }
                ++r;
            }
            sum += (Scalar) match_as_int;
        }
        return sum / Y_hat->cols();
    }


    void generate_random_weights(const vector<unsigned short> nodes_per_layer,
                                vector<shared_ptr<Matrix>>& weights,
                                vector<shared_ptr<ColVector>>& constants)
    {

        srand((unsigned int) time(0));
        
        weights =       vector<shared_ptr<Matrix>>(nodes_per_layer.size() - 1);
        constants =     vector<shared_ptr<ColVector>>(nodes_per_layer.size() - 1);

        for (size_t i = 1; i < nodes_per_layer.size(); i++) {
            constants[i-1] = make_shared<ColVector>(nodes_per_layer[i], 1);
            constants[i-1]->setRandom();
            (*constants[i-1]) *= 0.1;
            weights[i-1] = make_shared<Matrix>(nodes_per_layer[i], nodes_per_layer[i-1]);
            weights[i-1]->setRandom();
            (*weights[i-1]) *= 0.1;
        }
    }

    void generate_random_weights(const vector<unsigned short> nodes_per_layer,
                                vector<Matrix>& weights,
                                vector<ColVector>& constants)
    {
        weights =       vector<Matrix>(nodes_per_layer.size() - 1);
        constants =     vector<ColVector>(nodes_per_layer.size() - 1);

        for (size_t i = 1; i < nodes_per_layer.size(); i++) {
            constants[i-1] = ColVector(nodes_per_layer[i], 1);
            constants[i-1].setRandom();
            (constants[i-1]) *= 0.1;
            weights[i-1] = Matrix(nodes_per_layer[i], nodes_per_layer[i-1]);
            weights[i-1].setRandom();
            (weights[i-1]) *= 0.1;
        }
    }

}