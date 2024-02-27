---
Last Updated: 02-26-2024 | 6:24 PM
---
Gradient descent is an [[optimization algorithm]] used iteratively to find the minimum of a function by moving in the negative direction of the gradient of a function.

In the context of [[deep learning]], the function being minimized is the [[loss function]].
This allows for a model to update its weights and biases in the direction what yields a minimal value for a [[loss function]], ultimately minimizing inaccuracy in a [[neural network]].

It does so through calculating $\frac{d(loss)}{d(param)}$, to figure out where to optimize towards.

Say we have a dummy [[loss function]]: $y = x^2$, parameter: $x_0 = 4$, and a learning rate: $α =.05$

1. Find the gradient (derivative) of the function

	$y = x^2$
	$\frac{dy}{dx} = 2x$

2. Perform an iteration of gradient descent.

	$x_{1}=x_{0}-α·\frac{dy}{dx}$ 
	$x_{1}=4-.05·8$
	$x_{1}=4-.4$
	$x_{1}=3.6$

$3.6$ is the updated parameter for the next training pass.

This value will be then used as the starting point for future uses of the gradient descent algorithm.