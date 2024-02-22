---
created: 2024-02-18T15:31
Last Updated: 02-19-2024 | 11:28 AM
---
An activation function is how the weighted sum of the input is transferred into a discernible output for the next layer.

The choice of a specific activation functions can determine how the network ultimately performs its task. 

Different activation functions can be used in different parts of a network, though within a given layer, each neuron uses the same activation function.

It's applied to the weighted sum of a given neuron.

For example, If [[weighted sum]] is $z$ and we want to use [[ReLU]],

$a_i = ReLU(z_i)$

where a is the output of $ith$ [[weighted sum]] applied onto [[ReLU]]

Activation Functions:
- [[Sigmoid]]
- [[tanh]]
- [[ReLU]]
- [[Leaky ReLU]]
- [[SoftMax]]