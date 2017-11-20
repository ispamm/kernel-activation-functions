# -*- coding: utf-8 -*-

"""
Simple demo using kernel activation functions with convolutional layers on the MNIST dataset.
"""

# Imports from Python libraries
import numpy as np
from sklearn import datasets, preprocessing, model_selection

# PyTorch imports
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
import torch.utils.data
from torchvision import datasets, transforms
from torch.nn import Module

# Custom imports
from kafnets import KAF, KAF2D

# Set seed for PRNG
np.random.seed(1)
torch.manual_seed(1)

# Enable CUDA (optional)
enable_CUDA = True
if enable_CUDA:
    torch.cuda.device(0)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True # Comment for smaller convnets

# Batch size
B = 40

# Load MNIST dataset
kwargs = {'num_workers': 1, 'pin_memory': True} if (enable_CUDA & torch.cuda.is_available()) else {}
train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/MNIST', train=True, download=True,
                                                          transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=32, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/MNIST', train=False, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=32, shuffle=True, **kwargs)

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
    KAF(20),
    torch.nn.Conv2d(20, 20, kernel_size=5, padding=(2,2)),
    torch.nn.MaxPool2d(3),
    KAF(20),
    Flatten(),
    torch.nn.Linear(180, 10),
    torch.nn.LogSoftmax()
)

# Uncomment to use 2D-KAFs
# kafnet = torch.nn.Sequential(
#     torch.nn.Conv2d(1, 20, kernel_size=5, padding=(2,2)),
#     torch.nn.MaxPool2d(3),
#     KAF2D(20),
#     torch.nn.Conv2d(10, 20, kernel_size=5, padding=(2,2)),
#     torch.nn.MaxPool2d(3),
#     KAF2D(20),
#     Flatten(),
#     torch.nn.Linear(90, 10),
#     torch.nn.LogSoftmax()
# )

# Reset parameters
for m in kafnet:
    if len(m._parameters) > 0:
        m.reset_parameters()

print('\tTraining: **KAFNET**', flush=True)

# Loss function
loss_fn = torch.nn.NLLLoss(size_average=True)

# Build optimizer
optimizer = torch.optim.Adam(kafnet.parameters(), weight_decay=1e-4)

# Put model on GPU if needed
if enable_CUDA:
    print('\tMoving model to GPU...\n', flush=True)
    kafnet.cuda()

max_epochs = 10
for idx_epoch in range(max_epochs):

    print('Epoch #', idx_epoch, ' of #', max_epochs)
    kafnet.train()

    for _, (X_batch, y_batch) in enumerate(train_loader):

        # Eventually move mini-batch to GPU
        if enable_CUDA:
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

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
for _, (X_batch, y_batch) in enumerate(test_loader):
    # Eventually move mini-batch to GPU
    if enable_CUDA:
        X_batch = X_batch.cuda()
    acc += np.sum(y_batch.numpy() == np.argmax(kafnet(Variable(X_batch)).data.cpu().numpy(), axis=1))
print('\t\tFinal score on test set: ', acc / test_loader.dataset.__len__())
