# -*- coding: utf-8 -*-

"""
Simple demo using kernel activation functions on a basic classification dataset.
"""

# Imports from Python libraries
import numpy as np
from sklearn import datasets, preprocessing, model_selection

# PyTorch imports
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Custom imports
from kafnets import KAF, KAF2D

# Set seed for PRNG
np.random.seed(1)
torch.manual_seed(1)

# Batch size
B = 40

# Load Breast Cancer dataset
data = datasets.load_breast_cancer()
X = preprocessing.MinMaxScaler(feature_range=(-1, +1)).fit_transform(data['data']).astype(np.float32)
(X_train, X_test, y_train, y_test) = model_selection.train_test_split(X, data['target'].astype(np.float32).reshape(-1, 1), test_size=0.25)
data_train = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), shuffle=True, batch_size=64)
data_test = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=100)

# Initialize a KAF neural network
kafnet = torch.nn.Sequential(
    torch.nn.Linear(30, 20),
    KAF(20),
    torch.nn.Linear(20, 1),
)

# Uncomment to use 2D-KAF
#kafnet = torch.nn.Sequential(
#    torch.nn.Linear(30, 20),
#    KAF2D(20),
#    torch.nn.Linear(10, 1),
#)

# Reset parameters
for m in kafnet:
    if len(m._parameters) > 0:
        m.reset_parameters()

print('\tTraining: **KAFNET**', flush=True)

# Loss function
loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)

# Build optimizer
optimizer = torch.optim.Adam(kafnet.parameters(), weight_decay=1e-4)

for idx_epoch in range(100):

    kafnet.train()

    for _, (X_batch, y_batch) in enumerate(data_train):

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = kafnet(Variable(X_batch))

        # Compute loss.
        loss = loss_fn(y_pred, Variable(y_batch, requires_grad=False))

        # Zeroes out all gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

# Compute final test score
print('\n\t\tComputing test score for: **KAFNET**', flush=True)
kafnet.eval()
acc = 0
for _, (X_batch, y_batch) in enumerate(data_test):
    acc += np.sum(y_batch.numpy() == np.round(torch.sigmoid(kafnet(Variable(X_batch))).data.numpy()))
print('\t\tFinal score on test set: ', acc/data_test.dataset.__len__())
