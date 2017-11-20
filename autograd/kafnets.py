# -*- coding: utf-8 -*-

import autograd.numpy as np

###############################################################################
### KAF NEURAL NETWORK ########################################################
###############################################################################

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
    
    # Initialie a list of parameters
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
    

###############################################################################
### 2D KAF NEURAL NETWORK #####################################################
###############################################################################

def init_2d_kaf_nn(layer_sizes, scale=0.01, rs=np.random.RandomState(0), dict_size=10, boundary=3.0):
    """ 
    Initialize the parameters of a 2D-KAF feedforward network.
    """
    
    # Initialize the two-dimensional dictionary
    x = np.linspace(-boundary, boundary, dict_size).reshape(-1, 1)
    Dx, Dy = np.meshgrid(x, x)
    Dx = Dx.reshape(-1, 1)
    Dy = Dy.reshape(-1, 1)
    
    # Rule of thumb for gamma
    interval = x[1,0] - x[0,0];
    gamma = 0.5/np.square(2*interval/np.sqrt(2))
    
    # Halves the number of output neurons for every layer except the last one
    l = layer_sizes.copy()
    for i in np.arange(1, len(layer_sizes) - 1):
        l[i] = int(layer_sizes[i]/2)
    
    w = [(rs.randn(insize, outsize, 2) * scale,                 # Weight matrix
                     rs.randn(outsize, 2) * scale,              # Bias vector
                     rs.randn(1, outsize, Dx.shape[0]) * 0.5)   # Mixing coefficients
                     for insize, outsize in zip(l[:-1], l[1:])]
    
    return w, (Dx, Dy, gamma)

def predict_2d_kaf_nn(w, X, info):
    """
    Compute the outputs of a 2D-KAF feedforward network.
    """
    Dx, Dy, gamma = info
    for W, b, alpha in w:
        outputs = np.tensordot(X, W, axes=1) + b
        K = gauss_2d_kernel(outputs, (Dx, Dy), gamma)
        X = np.sum(K*alpha, axis=2)
    return X


def gauss_2d_kernel(X, D, gamma=1.0):
    """
    Compute the 2D Gaussian kernel between all elements of a 
    NxHx2 matrix and a fixed L-dimensional dictionary, resulting in a NxHxL matrix of kernel
    values.
    """
    (Dx, Dy) = D
    return np.exp(-gamma*np.square(X[:,:,0].reshape(-1, X.shape[1], 1) - Dx.reshape(1, 1, -1)) \
                  -gamma*np.square(X[:,:,1].reshape(-1, X.shape[1], 1) - Dy.reshape(1, 1, -1)))