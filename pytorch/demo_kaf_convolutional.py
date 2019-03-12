# -*- coding: utf-8 -*-

"""
Simple demo using kernel activation functions with convolutional layers on the MNIST dataset.
"""

# Imports from Python libraries
import numpy as np
import tqdm

# PyTorch imports
import torch
import torch.utils.data
from torchvision import datasets, transforms
from torch.nn import Module

# Custom imports
from kafnets import KAF

# Set seed for PRNG
np.random.seed(1)
torch.manual_seed(1)

# Enable CUDA (optional)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load MNIST dataset
train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/MNIST', train=True, download=True,
                                                          transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/MNIST', train=False, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=32, shuffle=True)

class Flatten(Module):
    """
    Simple flatten module, see this discussion:
    https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

# Initialize a KAF neural network
kafnet = torch.nn.Sequential(
    torch.nn.Conv2d(1, 20, kernel_size=5, padding=(2,2)),
    torch.nn.MaxPool2d(3),
    KAF(20, conv=True),
    torch.nn.Conv2d(20, 20, kernel_size=5, padding=(2,2)),
    torch.nn.MaxPool2d(3),
    KAF(20, conv=True),
    Flatten(),
    torch.nn.Linear(180, 10),
)

# Reset parameters
for m in kafnet:
    if len(m._parameters) > 0:
        m.reset_parameters()

print('Training: **KAFNET**', flush=True)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Build optimizer
optimizer = torch.optim.Adam(kafnet.parameters(), weight_decay=1e-4)

# Put model on GPU if needed
kafnet.to(device)

max_epochs = 10
for idx_epoch in range(max_epochs):

    print('Epoch #', idx_epoch, ' of #', max_epochs)
    kafnet.train()

    for (X_batch, y_batch) in tqdm.tqdm(train_loader):

        # Eventually move mini-batch to GPU
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

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

# Compute final test score
with torch.no_grad():
    print('Computing test score for: **KAFNET**', flush=True)
    kafnet.eval()
    acc = 0
    for _, (X_batch, y_batch) in enumerate(test_loader):
        # Eventually move mini-batch to GPU
        X_batch = X_batch.to(device)
        acc += np.sum(y_batch.numpy() == np.argmax(kafnet(X_batch).cpu().numpy(), axis=1))
    print('Final score on test set: ', acc / test_loader.dataset.__len__())
