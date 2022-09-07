import numpy as np

from activation_functions import *


def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = (Y * np.log(Y_hat) + (1 - Y) *  np.log(1 - Y_hat))
    return - cost.sum() / m



def get_accuracy_value(Y_hat, Y):
    # convert the estimates Y_hat into vectors of 1 and 0
    Y_hat_ = np.copy(Y_hat)
    Y_hat_[Y_hat_ > 0.5] = 1
    Y_hat_[Y_hat_ <= 0.5] = 0
    # count the number of 0, 1 vectors that are the same
    return (Y_hat_ == Y).all(axis=0).mean()


def get_random_weights(nodes_per_layer, seed = 42):
    np.random.seed(seed)
    parameters = {}
    for i in range(0, len(nodes_per_layer) - 1):
        parameters['W' + str(i)] = np.random.randn(nodes_per_layer[i+1], nodes_per_layer[i]) * 0.1
        parameters['b' + str(i)] = (np.random.randn(nodes_per_layer[i+1]) * 0.1).reshape(nodes_per_layer[i+1], 1)
    return parameters


def train(nodes_per_layer, learning_rate, parameters, X, Y, epochs, output_training_stats = True, output_stats_every_n_steps = 50):
    for epoch in range(0, epochs):
        memory = {}
        neuron_values_curr = X
        # forward propagation    
        for i in range(0, len(nodes_per_layer) - 1):
            neuron_layer_prev = neuron_values_curr
            weight_curr = parameters["W" + str(i)]
            constants_curr = parameters["b" + str(i)]
            unactivated_values_curr = np.dot(weight_curr, neuron_layer_prev) + constants_curr           
            neuron_values_curr = sigmoid(unactivated_values_curr)
            memory["A" + str(i)] = neuron_layer_prev
            memory["Z" + str(i+1)] = unactivated_values_curr

        # backwards propagation
        grads_values = {}
        m = Y.shape[1]
        Y = Y.reshape(neuron_values_curr.shape)
        # derivative of logistic regression function \mathscr{L} = - y / y_hat + (1-y) / (1-y_hat)
        d_neuron_layer_prev = - (np.divide(Y, neuron_values_curr) - np.divide(1 - Y, 1 - neuron_values_curr));        
        for i in reversed(range(0, len(nodes_per_layer) - 1)):
            d_neuron_values_curr = d_neuron_layer_prev
            neuron_layer_prev = memory["A" + str(i)]
            unactivated_values_curr = memory["Z" + str(i+1)]
            weight_curr = parameters["W" + str(i)]
            constants_curr = parameters["b" + str(i)]
            m = neuron_layer_prev.shape[1]       
            d_unactivated_values_curr = d_neuron_values_curr * sigmoid_backward(unactivated_values_curr)
            d_weight_curr = np.dot(d_unactivated_values_curr, neuron_layer_prev.T) / m
            d_constants_curr = np.sum(d_unactivated_values_curr, axis=1, keepdims=True) / m
            if (i>0): # Don't need this on the last step and it has a big performance impact
                d_neuron_layer_prev = np.dot(weight_curr.T, d_unactivated_values_curr)

            grads_values["dW" + str(i)] = d_weight_curr
            grads_values["db" + str(i)] = d_constants_curr

        # update parameters
        for i in range(0, len(nodes_per_layer) - 1):
            parameters["W" + str(i)] -= learning_rate * grads_values["dW" + str(i)]
            parameters["b" + str(i)] -= learning_rate * grads_values["db" + str(i)]

        if output_training_stats and (epoch % output_stats_every_n_steps == 0):
            cost = get_cost_value(neuron_values_curr, Y)
            accuracy = get_accuracy_value(neuron_values_curr, Y)
            print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(epoch, cost, accuracy))

    return parameters