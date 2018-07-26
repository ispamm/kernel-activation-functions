# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn.init import normal


class KAF(Module):
    """Basic KAF module. Each activation function is a kernel expansion over a fixed dictionary
    of D elements, and the combination coefficients are learned.
    """

    def __init__(self, num_parameters, D=20, boundary=3.0, init_fcn=None):
        """
        :param num_parameters: number of neurons in the layer.
        :param D: size of the dictionary.
        :param boundary: range of the activation function.
        :param init_fcn: leave to None to initialize randomly, otherwise set a specific function for initialization.
        """
        self.num_parameters, self.D = num_parameters, D
        super(KAF, self).__init__()

        # Initialize the fixed dictionary
        self.dict_numpy = np.linspace(-boundary, boundary, self.D).astype(np.float32).reshape(-1, 1)
        self.register_buffer('dict', torch.from_numpy(self.dict_numpy).view(-1))

        # Rule of thumb for gamma
        interval = (self.dict[1] - self.dict[0])
        sigma = 2 * interval  # empirically chosen
        self.gamma = 0.5 / np.square(sigma)

        # Initialization
        self.init_fcn = init_fcn
        if init_fcn is not None:
            # Initialization with kernel ridge regression
            K = np.exp(- self.gamma*(self.dict_numpy - self.dict_numpy.T) ** 2)
            self.alpha_init = np.linalg.solve(K + 1e-5*np.eye(self.num_parameters), self.init_fcn(self.dict_numpy)).reshape(-1)
        else:
            # Random initialization
            self.alpha_init = None

        # Mixing coefficients
        self.alpha = Parameter(torch.FloatTensor(1, self.num_parameters, self.D))

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_fcn is not None:
            self.alpha.data = torch.from_numpy(self.alpha_init).repeat(1, self.num_parameters, 1)
        else:
            normal(self.alpha.data, std=0.3)

    def forward(self, input):
        # Flatten dimensions
        s = input.size()
        if len(s) == 4: # Convolutional
            K = torch.exp(- torch.mul((torch.add(input.unsqueeze(4), - Variable(self.dict))) ** 2, self.gamma))
            y = torch.sum(K * self.alpha.view(1, self.num_parameters, 1, 1, self.D), 4)
        else:
            # First computes the Gaussian kernel
            K = torch.exp(- torch.mul((torch.add(input.unsqueeze(2), - Variable(self.dict)))**2, self.gamma))
            y = torch.sum(K*self.alpha, 2)
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.num_parameters) + ')'


class KAF2D(Module):
    """Basic 2D-KAF module. Each activation function is a kernel expansion over a pair
    of activation values.
    """

    def __init__(self, num_parameters, D=10, boundary=3.0):
        """
        :param num_parameters: number of neurons (gets halved in output).
        :param D: size of the dictionary.
        :param boundary: range of the activation functions.
        """
        super(KAF2D, self).__init__()
        self.num_parameters, self.D = num_parameters, D

        # Check that the number of parameters is even
        if np.mod(self.num_parameters, 2) != 0:
            raise ValueError('The number of parameters for KAF2D must be even.')

        # Saves the actual number of output values
        self.num_parameters = int(self.num_parameters / 2)

        # Initialize the fixed dictionary
        x = np.linspace(-boundary, boundary, self.D).astype(np.float32).reshape(-1, 1)
        Dx, Dy = np.meshgrid(x, x)
        self.register_buffer('Dx', torch.from_numpy(Dx.reshape(1, 1, -1)))
        self.register_buffer('Dy', torch.from_numpy(Dy.reshape(1, 1, -1)))

        # Rule of thumb for gamma
        interval = (x[1, 0] - x[0, 0])
        sigma = 2*interval/np.sqrt(2)  # empirically chosen
        self.gamma = 0.5/np.square(sigma)

        # Mixing coefficients
        self.alpha = Parameter(torch.FloatTensor(1, self.num_parameters, self.D*self.D))

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        normal(self.alpha.data, std=0.3)

    def gauss_2d_kernel(self, X, s=None):
        if s is None:
            s = X.size()
        if len(s) == 4: # Convolutional
            tmp = -torch.mul((torch.add(X[:, 0:self.num_parameters].unsqueeze(4), - Variable(self.Dx).view(1, 1, 1, 1, -1))) ** 2, self.gamma) - \
                  torch.mul((torch.add(X[:, self.num_parameters:].unsqueeze(4), - Variable(self.Dy).view(1, 1, 1, 1, -1))) ** 2, self.gamma)
        else:
            tmp = -torch.mul((torch.add(X[:, 0:self.num_parameters].unsqueeze(2), - Variable(self.Dx)))**2, self.gamma) - \
                  torch.mul((torch.add(X[:, self.num_parameters:].unsqueeze(2), - Variable(self.Dy))) ** 2, self.gamma)
        return torch.exp(tmp)

    def forward(self, input):
        s = input.size()
        # Compute the 2D Gaussian kernel
        K = self.gauss_2d_kernel(input, s)
        if len(s) == 4:
            y = torch.sum(K * self.alpha.view(1, -1, 1, 1, self.D * self.D), 4)
        else:
            y = torch.sum(K * self.alpha, 2)
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.num_parameters) + ')'
