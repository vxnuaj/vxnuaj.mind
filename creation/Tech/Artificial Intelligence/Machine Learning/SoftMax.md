---
Last Updated: 03-09-2024 | 5:27 PM
---
The SoftMax [[activation function]] is commonly used in the output layer of a neural network for multi-class classification.

![[softmax.png|500]]

It takes in a vector of [[logits]] and computes a probability for each class of the dataset.

For example, say you have final layer of 10 output neurons, each with a weighted sum $z_i$, where $i$ represents $ith$ neuron, and 10 total classes indexed by $j$

You take a vector of all weighted sums $z_i$,

$z = [z_1, z_2, z_3 ... z_{10}]$

and feed it to the SoftMax activation function,

$\hat y_{i}=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}$

to ultimately get a probability for each value $z_i$
