# Kernel Activation Functions

This repository contains several implementations of the kernel activation functions (KAFs) described in the following paper ([link to the preprint](https://arxiv.org/abs/1707.04035)):
	
	Scardapane, S., Van Vaerenbergh, S., Totaro, S. and Uncini, A., 2019. 
	Kafnets: Kernel-based non-parametric activation functions for neural networks. 
	Neural Networks, 110, pp.19-32.
	
## Available implementations

We currently provide the following stable implementations:

* [PyTorch](/pytorch): feedforward and convolutional networks, three kernels (Gaussian/ReLU/Softplus), with random initialization or kernel ridge regression.
* [Keras](/keras): same as the PyTorch implementation.
* [Autograd](/autograd): only for feedforward networks with Gaussian kernel and random initialization.

The following implementations are not stable and are under development:

* [TensorFlow](/tensorflow/): feedforward and convolutional networks.
	
More information for each implementation is given in the corresponding folder. The code should be relatively easy to plug-in in other architectures or projects.

## What is a KAF?

Most neural networks work by interleaving linear projections and simple (fixed) activation functions, like the ReLU function:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?g(s)&space;=&space;\max&space;\left(&space;0,&space;s&space;\right&space;\)&space;\,." title="g(s) = \max \left( 0, s \right \) \,." />
</p>

A KAF is instead a non-parametric activation function defined as a one-dimensional kernel approximator:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?g(s)&space;=&space;\sum_{i=1}^D&space;\alpha_i&space;\kappa\left(s,&space;d_i\right)&space;\,," title="g(s) = \sum_{i=1}^D \alpha_i \kappa\left(s, d_i\right) \,," />
</p>

where:

* The dictionary of the kernel elements is fixed by sampling the x-axis with a uniform step around 0.
* The user can select the kernel function (e.g., Gaussian, ReLU, Softplus) and the number of kernel elements D.
* The linear coefficients are adapted independently at every neuron via standard back-propagation.

In addition, the linear coefficients can be initialized using kernel ridge regression to behave similarly to a known function in the beginning of the optimization process.

## Contributing

If you have an implementation for a different framework, or an enhanced version of the current code, feel free to contribute to the repository. For any issues related to the code you can use the issue tracker from GitHub.

## Citation

If you use this code or a derivative thereof in your research, we would appreciate a citation to the original paper:

	@article{scardapane2019kafnets,
      title={Kafnets: Kernel-based non-parametric activation functions for neural networks},
      author={Scardapane, Simone and Van Vaerenbergh, Steven and Totaro, Simone and Uncini, Aurelio},
      journal={Neural Networks},
      volume={110},
      pages={19--32},
      year={2019},
      publisher={Elsevier}
    }
	
## License

The code is released under the MIT License. See the attached LICENSE file.
