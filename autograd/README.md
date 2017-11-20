## Kernel activation functions (Autograd)

The code customizes the example from here:
https://github.com/HIPS/autograd/blob/master/examples/neural_net.py

In the *kafnets* module you can find the code to initialize and run neural networks having KAFs or 2D-KAFs activation functions. Note that this is a regression example and we use custom functions also in the output layer. For classification, consider changing the final layer to a standard softmax.

## Requirements

The code has been tested with Python v3.6 and Autograd v1.2.