One-hot encoding transforms the labels of a dataset into vectors of 0s and 1s.

These vectors have a length equal to the total number of classes or categories that a model is expected to classify.

**As an example,** in the [[Neural Network for MNIST]], the length of these encoded vectors would be 10, given that we're classifying numbers 0-9.

Each index of that vector corresponds to 1 of the 9 digits represented in the dataset.

A vector for the digit 2, may look as such: $[0,0,1,0,0,0,0,0,0,0]$

