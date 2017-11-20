# Kernel Activation Functions

This repository contains several implementations of the kernel activation functions (KAFs) described in the following preprint:
	
	Scardapane, S., Van Vaerenbergh, S., Totaro, S. and Uncini, A., 2017. Kafnets: kernel-based non-parametric activation functions for neural networks. arXiv preprint arXiv:1707.04035.
	
## Available implementations

We currently provide the following implementations:

	* *PyTorch* implementation of KAF, 2D-KAF, both for feedforward and convolutional networks.
	* *TensorFlow* implementation of KAF, only for feedforward networks.
	* *Autograd* implementation of KAF, 2D-KAF, only for feedforward networks.
	
More information for each implementation is given in the corresponding folder. Most of the code should be easy to plug-in in other architectures.

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
