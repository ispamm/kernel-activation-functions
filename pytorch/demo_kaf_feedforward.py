# -*- coding: utf-8 -*-

"""
Simple demo using kernel activation functions on a basic classification dataset.
"""

# Imports from Python libraries
import numpy as np
from sklearn import datasets, preprocessing, model_selection

# PyTorch imports
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Custom imports
from kafnets import KAF

# Set seed for PRNG
np.random.seed(1)
torch.manual_seed(1)

# Batch size
B = 40

# Load Breast Cancer dataset
data = datasets.load_breast_cancer()
X = preprocessing.MinMaxScaler(feature_range=(-1, +1)).fit_transform(data['data']).astype(np.float32)
(X_train, X_test, y_train, y_test) = model_selection.train_test_split(X, data['target'].astype(np.float32).reshape(-1, 1), test_size=0.25)

# Load in PyTorch data loader
data_train = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), shuffle=True, batch_size=64)
data_test = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=100)

# Initialize a KAF neural network
kafnet = torch.nn.Sequential(
    torch.nn.Linear(30, 20),
    KAF(20),
    torch.nn.Linear(20, 1),
)

# Uncomment to use KAF with custom initialization
#kafnet = torch.nn.Sequential(
#    torch.nn.Linear(30, 20),
#    KAF(20, init_fcn=np.tanh),
#    torch.nn.Linear(20, 1),
#)

#Uncomment to use KAF with Softplus kernel
#kafnet = torch.nn.Sequential(
#    torch.nn.Linear(30, 20),
#    KAF(20, kernel='softplus'),
#    torch.nn.Linear(20, 1),
#)

# Reset parameters
for m in kafnet:
    if len(m._parameters) > 0:
        m.reset_parameters()

print('Training: **KAFNET**', flush=True)

# Loss function
loss_fn = torch.nn.BCEWithLogitsLoss()

# Build optimizer
optimizer = torch.optim.Adam(kafnet.parameters(), weight_decay=1e-4)

for idx_epoch in range(100):

    kafnet.train()

    for _, (X_batch, y_batch) in enumerate(data_train):

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = kafnet(X_batch)

        # Compute loss.
        loss = loss_fn(y_pred, y_batch)

        # Zeroes out all gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

with torch.no_grad():
    # Compute final test score
    print('Computing test score for: **KAFNET**', flush=True)
    kafnet.eval()
    acc = 0
    for _, (X_batch, y_batch) in enumerate(data_test):
        acc += np.sum(y_batch.numpy() == np.round(torch.sigmoid(kafnet(X_batch)).numpy()))
    print('Final score on test set: ', acc/data_test.dataset.__len__())
