# -*- coding: utf-8 -*-

import autograd.numpy as np

def init_kaf_nn(layer_sizes, scale=0.01, rs=np.random.RandomState(0), dict_size=20, boundary=3.0):
    """ 
    Initialize the parameters of a KAF feedforward network.
        - dict_size: the size of the dictionary for every neuron.
        - boundary: the boundary for the activation functions.
    """
    
    # Initialize the dictionary
    D = np.linspace(-boundary, boundary, dict_size).reshape(-1, 1)
    
    # Rule of thumb for gamma
    interval = D[1,0] - D[0,0];
    gamma = 0.5/np.square(2*interval)
    D = D.reshape(1, 1, -1)
    
    # Initialize a list of parameters for the layer
    w = [(rs.randn(insize, outsize) * scale,                # Weight matrix
                     rs.randn(outsize) * scale,             # Bias vector
                     rs.randn(1, outsize, dict_size) * 0.5) # Mixing coefficients
                     for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]
    
    return w, (D, gamma)

def predict_kaf_nn(w, X, info):
    """
    Compute the outputs of a KAF feedforward network.
    """
    
    D, gamma = info
    for W, b, alpha in w:
        outputs = np.dot(X, W) + b
        K = gauss_kernel(outputs, D, gamma)
        X = np.sum(K*alpha, axis=2)
    return X

def gauss_kernel(X, D, gamma=1.0):
    """
    Compute the 1D Gaussian kernel between all elements of a 
    NxH matrix and a fixed L-dimensional dictionary, resulting in a NxHxL matrix of kernel
    values.
    """
    return np.exp(- gamma*np.square(X.reshape(-1, X.shape[1], 1) - D))