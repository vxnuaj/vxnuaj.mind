---
created: 2024-02-19T13:03
Last Updated: 02-20-2024 | 9:33
---
One-hot encoding transforms the labels of a dataset into vectors of 0s and 1s, to allow for a model to more easily interpret & then classify pieces of data.

These vectors have a length equal to the total number of classes that a model is expected to classify.

For each one-hot encoded vector, each element will be a 0 except for the element that identifies the specific identified class.

**As an example,** in the [[Neural Network for MNIST]], the length of these encoded vectors would be 10, given that we're classifying numbers 0-9.

Each element of that vector corresponds to 1 of the 0-9 possible digits in their respective order.

A vector for digit 2, may look as such: $[0,0,1,0,0,0,0,0,0,0]$
Or a vector for digit 8, may look as such: $[0,0,0,0,0,0,0,0,1,0]$

