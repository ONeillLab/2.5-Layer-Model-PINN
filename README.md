
Work-in-progress neural networks for simulating the 2.5 layer shallow water model.

This system is very nonlinear, and so is not amenable to basic machine learning techniques. 

The goal is to successfully learn on a small, coarse resolution grid. Higher resolution can then be achieved by traditional parallelization. This should exponentially decrease computation time. 

I've so far tried a physics-informed neural network, a hybrid numerical-physics-informed neural network, and a graph neural network with architecture similar to GraphCast
