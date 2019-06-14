import numpy as np
from keras.layers import Layer
from keras import backend as K

class KAF(Layer):
    """ Implementation of the kernel activation function.
    
    Parameters
    ----------
    num_parameters: int
        Size of the layer (number of neurons).
    D: int, optional
        Size of the dictionary for each neuron. Default to 20.
    conv: bool, optional
        True if this is a convolutive layer, False for a feedforward layer. Default to False.
    boundary: float, optional
        Dictionary elements are sampled uniformly in [-boundary, boundary]. Default to 4.0.
    init_fcn: None or func, optional
        If None, elements are initialized randomly. Otherwise, elements are initialized to approximate given function.
    kernel: {'gauss', 'relu', 'softplus'}, optional
        Kernel function to be used. Defaults to 'gaussian'.
    
    Example
    ----------
    Neural network with one hidden layer with KAF nonlinearities:
        
    >>> net = Sequential([Dense(10), KAF(10), Dense(10, 1)])
    
    References
    ----------
    [1] Scardapane, S., Van Vaerenbergh, S., Totaro, S. and Uncini, A., 2019. 
        Kafnets: kernel-based non-parametric activation functions for neural networks. 
        Neural Networks, 110, pp. 19-32.
    [2] Marra, G., Zanca, D., Betti, A. and Gori, M., 2018. 
        Learning Neuron Non-Linearities with Kernel-Based Deep Neural Networks. 
        arXiv preprint arXiv:1807.06302.
    """

    def __init__(self, num_parameters, D=20, boundary=3.0, conv=False, init_fcn=None, kernel='gaussian', **kwargs):
        self.num_parameters = num_parameters
        self.D = D
        self.boundary = boundary
        self.init_fcn = init_fcn
        self.conv = conv
        if self.conv:
            self.unsqueeze_dim = 4
        else:
            self.unsqueeze_dim = 2
        self.kernel = kernel
        if not (kernel in ['gaussian', 'relu', 'softplus']):
            raise ValueError('Kernel not recognized (must be {gaussian, relu, softplus})')
        super().__init__(**kwargs)
    
    def build(self, input_shape):

        # Initialize the fixed dictionary
        d = np.linspace(-self.boundary, self.boundary, self.D).astype(np.float32).reshape(-1, 1)
        
        if self.conv:
            self.dict = self.add_weight(name='dict', 
                                      shape=(1, 1, 1, 1, self.D),
                                      initializer='uniform',
                                      trainable=False)
            K.set_value(self.dict, d.reshape(1, 1, 1, 1, -1))
        else:
            self.dict = self.add_weight(name='dict', 
                                      shape=(1, 1, self.D),
                                      initializer='uniform',
                                      trainable=False)
            K.set_value(self.dict, d.reshape(1, 1, -1))
        
        if self.kernel == 'gaussian':
            self.kernel_fcn = self.gaussian_kernel
            # Rule of thumb for gamma
            interval = (d[1] - d[0])
            sigma = 2 * interval  # empirically chosen
            self.gamma = 0.5 / np.square(sigma)
        elif self.kernel == 'softplus':
            self.kernel_fcn = self.softplus_kernel
        else:
            self.kernel_fcn = self.relu_kernel
            
        
        # Mixing coefficients
        if self.conv:
            self.alpha = self.add_weight(name='alpha', 
                                         shape=(1, 1, 1, self.num_parameters, self.D),
                                         initializer='normal',
                                         trainable=True)
        else:
            self.alpha = self.add_weight(name='alpha', 
                                         shape=(1, self.num_parameters, self.D),
                                         initializer='normal',
                                         trainable=True)

        # Optional initialization with kernel ridge regression
        if self.init_fcn is not None:
            if self.kernel == 'gaussian':
              kernel_matrix = np.exp(- self.gamma*(d - d.T) ** 2)
            elif self.kernel == 'softplus':
              kernel_matrix = np.log(np.exp(d - d.T) + 1.0)
            else:
              raise ValueError('Cannot perform kernel ridge regression with ReLU kernel (singular matrix)')
            
            alpha_init = np.linalg.solve(kernel_matrix + 1e-5*np.eye(self.D), self.init_fcn(d)).reshape(-1)
            if self.conv:
                K.set_value(self.alpha, np.repeat(alpha_init.reshape(1, 1, 1, 1, -1), self.num_parameters, axis=3))
            else:
                K.set_value(self.alpha, np.repeat(alpha_init.reshape(1, 1, -1), self.num_parameters, axis=1))
        
        super(KAF, self).build(input_shape)
        
    def gaussian_kernel(self, x):
        return K.exp(- self.gamma * (K.expand_dims(x, axis=self.unsqueeze_dim) - self.dict) ** 2.0)
        
    def softplus_kernel(self, x):
        return K.softplus(K.expand_dims(x, axis=self.unsqueeze_dim) - self.dict)
    
    def relu_kernel(self, x):
        return K.relu(K.expand_dims(x, axis=self.unsqueeze_dim) - self.dict)
    
    def call(self, x):
        kernel_matrix = self.kernel_fcn(x)
        return K.sum(kernel_matrix * self.alpha, axis=self.unsqueeze_dim)
    
    def get_config(self):
      return {'num_parameters': self.num_parameters,
             'D': self.D,
             'boundary': self.boundary,
             'conv': self.conv,
             'init_fcn': self.init_fcn,
             'kernel': self.kernel
      }
