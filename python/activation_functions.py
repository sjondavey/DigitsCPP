import numpy as np

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def sigmoid_backward(Z):
    sig = sigmoid(Z)
    return sig * (1 - sig)