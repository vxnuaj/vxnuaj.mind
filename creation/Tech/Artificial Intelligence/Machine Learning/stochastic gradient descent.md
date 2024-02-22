---
Last Updated: 02-21-2024 | 9:12 PM
---
Stochastic gradient descent (SGD) is a variation of traditional [[gradient descent]], also aiming to minimize the [[loss]] of the [[loss function]].

The mathematical principles between SGD and [[gradient descent]] are the same.

The difference is in how each algorithm updates the model parameters.

SGD updates the parameters after computing the gradient for each training sample individually. Gradient descent does the opposite, computing the gradient after running all training samples through a model.

This difference is expressed within the code of a model. It defines how the algorithm is applied differently, within each training sample, rather than training set.



 