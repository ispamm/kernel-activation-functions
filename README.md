# Kernel Activation Functions

This repository contains several implementations of the kernel activation functions (KAFs) described in the following preprint ([link to the arXiv page](https://arxiv.org/abs/1707.04035)):
	
	Scardapane, S., Van Vaerenbergh, S., Totaro, S. and Uncini, A., 2017. Kafnets: kernel-based 
	non-parametric activation functions for neural networks. arXiv preprint arXiv:1707.04035.
	
## Available implementations

We currently provide the following implementations:

* [PyTorch](/pytorch): KAF and 2D-KAF, both for feedforward and convolutional networks.
* [TensorFlow](/tensorflow/): KAF and 2D-KAF, both for feedforward and convolutional networks.
* [Autograd](/autograd/): KAF and 2D-KAF, only for feedforward networks.
	
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
* The user can select the kernel function (Gaussian in our implementation) and the number of kernel elements D.
* The linear coefficients are adapted independently at every neuron via standard back-propagation.

A two-dimensional KAF is defined similarly, but it can nonlinearly combine two different activation values. See [the original paper](https://arxiv.org/abs/1707.04035) for more details and some experimental results.

## Contributing

If you have an implementation for a different framework, or an enhanced version of the current code, feel free to contribute to the repository. For any issues related to the code you can use the issue tracker from GitHub.

## Citation

If you use this code or a derivative thereof in your research, we would appreciate a citation to the original paper:

	@article{scardapane2017kafnets,
		title={Kafnets: kernel-based non-parametric activation functions for neural networks},
		author={Scardapane, Simone and Van Vaerenbergh, Steven and Totaro, Simone and Uncini, Aurelio},
		journal={arXiv preprint arXiv:1707.04035},
		year={2017}
	}
	
## License

The code is released under the MIT License. See the attached LICENSE file.
