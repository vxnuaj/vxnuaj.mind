---
created: 2024-02-18T15:04
Last Updated: 02-24-2024 | 8:24 PM
---
Used during [[forward propagation]] to calculate the output of an individual neuron in a [[neural network]], it's the calculation of the $\sum$ of the multiplication of the [[weights]] and input with the addition of a [[bias]].

$z = \sum_{i=1}^{N}W_i*x_{i}+ b$,

where 
- $n$ is the total number of neurons
- $i$ is the index of each neuron
- $W_i$ is the [[weights]] at $ith$ index
- $b$ is the [[bias]] parameter

The output $z$ is applied to an [[activation function]] to get the final output. 