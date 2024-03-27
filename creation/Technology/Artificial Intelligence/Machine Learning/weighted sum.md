---
created: 2024-02-18T15:04
Last Updated: 03-09-2024 | 6:03 PM
---
Used during [[forward propagation]] to calculate the output of an individual neuron in a [[neural network]], it's the calculation of the $\sum$ of the multiplication of the [[weight]] and input with the addition of a [[bias]].

$z = \sum_{i=1}^{N}w_{ij}·x_{ij}+ b_i$,

where 
- $N$ is the total number of neurons
- $i$ is the index of each neuron in the current layer
- $j$ is the index of each input neuron from a previous layer into $ith$ neuron
- $w_{ij}$ is the [[weight]] at the connection between $jth$ and $ith$ neuron
- $x_{ij}$ is the input from the $jth$ neuron to the $ith$ neuron
- $b$ is the [[bias]] parameter at $ith$ neuron

The output $z$ is applied to an [[activation function]] to get the final output. 