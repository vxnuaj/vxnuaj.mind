---
Last Updated: 02-21-2024 | 2:30
---
Gradient descent is an [[optimization algorithm]] used iteratively to find the minimum of a function by moving in the negative direction of the gradient of a function.

In the context of deep learning, the function being minimized is the [[loss function]].
This then allows for a model to update its weights and biases in the direction what yields a minimal value for a [[loss function]], ultimately minimizing inaccuracy in a [[neural network]].

Say we have a [[loss function]]: $y = x^2$, initial output: $x_0 = 4$, and a learning rate: $α =.01$

1. Find the gradient (derivative) of the function

	$y = x^2$
	
	$\frac{dy}{dx} = 2x$

2. Perform an iteration of gradient descent.

	$x_{1}=x_{0}-α·\frac{dy}{dx}$
	
	$x_{1}=4-.01·8$ 
	
	$x_{1}=4-4$ 
	
	$x_{1}=4-4$ 