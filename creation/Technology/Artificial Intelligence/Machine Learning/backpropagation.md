---
created: 2024-02-19T12:17
Last Updated: 03-26-2024 | 10:05 PM
---
Backpropagation is used to minimize the output of the [[loss function ]]through adjusting the [[weight]]s and [[bias]]es of a given [[neural network]], through the *chain rule*, in order to increase the model accuracy.

It involves continually taking the gradient of a [[loss function]] with respect to a parameter (a weight or bias), of multiple layers, to ultimately allow for [[gradient descent]] or a similar optimization algorithm to minimize this gradient which then increases accuracy of a network.

The gradients of the loss w.r.t to the outputs of the output layer are calculated first and then propagated back to the subsequent layers to allow for the calculation of other parameters at $l$ layer, $i$ neuron (refer to [[Variables]]).
