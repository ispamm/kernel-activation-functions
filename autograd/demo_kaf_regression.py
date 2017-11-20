# -*- coding: utf-8 -*-

# Imports from Python libraries
import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
from sklearn import datasets, preprocessing, model_selection

# Custom imports
from kafnets import init_kaf_nn, predict_kaf_nn
from kafnets import init_2d_kaf_nn, predict_2d_kaf_nn

# Extends this example:
# https://github.com/HIPS/autograd/blob/master/examples/neural_net.py

# Set seed for PRNG
np.random.seed(1)

# Size of the neural network's layers
layers = [13, 10, 1]

# Batch size
B = 40

# Load Boston dataset
data = datasets.load_boston()
X = preprocessing.MinMaxScaler(feature_range=(-1, +1)).fit_transform(data['data'])
y = preprocessing.MinMaxScaler(feature_range=(-0.9, +0.9)).fit_transform(data['target'].reshape(-1, 1))
(X_train, X_test, y_train, y_test) = model_selection.train_test_split(X, y, test_size=0.25)

# Initialize KAF neural network
w, info = init_kaf_nn(layers)
predict_fcn = lambda w, inputs: predict_kaf_nn(w, inputs, info)

# Initialize 2D-KAF neural network (uncomment if needed)
# w, info = init_2d_kaf_nn(layers)
# predict_fcn = lambda w, inputs: predict_2d_kaf_nn(w, inputs, info)

# Loss function (MSE)
def loss_fcn(params, inputs, targets):
    return np.mean(np.square(predict_fcn(params, inputs) - targets))

# Iterator over mini-batches
num_batches = int(np.ceil(X_train.shape[0] / B))
def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * B, (idx+1) * B)

# Define training objective
def objective(params, iter):
    idx = batch_indices(iter)
    return loss_fcn(params, X_train[idx], y_train[idx])

# Get gradient of objective using autograd.
objective_grad = grad(objective)

# The optimizers provided can optimize lists, tuples, or dicts of parameters
print('Optimizing the network...\n')
w_final = adam(objective_grad, w, num_iters=1000)

# Compute test accuracy
print('Final test MSE is ', loss_fcn(w_final, X_test, y_test), '\n')