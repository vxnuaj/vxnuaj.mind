---
created: 2024-02-20T21:51
Last Updated: 02-20-2024 | 10:58
---
Categorical Cross Entropy Loss is a [[loss function]] which calculates the [[loss]] of a model's output.

It's specifically designed for multi-class classification tasks where each input belongs exactly to one class.

The equation is defined as

$CE = -\sum_{i=1}^{N}y_{i}·log(\hat{y_i})$

where,
- $\hat{y}_i$ is the prediction of a model as $ith$ class
- $y_i$ represents the one hot encoded vector of $ith$ class
- Index $i$ denotes the index of the $ith$ class of a dataset
- $N$ denotes the total number of classes in the dataset / [[one-hot encoding]] vector.

As an example:

If the value input into a model is 2, within a dataset of 3 classes $[0,1,2]$ and the prediction of the model is $[.1,.2,.7]$:

The equation for categorical cross entropy would be defined as such,

$CE =-\sum_{i=0}^{2}y_{i}·log(\hat{y}_i)$

And the calculation will look as,

$CE =-[(0⋅-1)+(0·-.699)+(1·-.155)]$
$CE =-[0+0-.155)]$

$CE ≈ .155$ 

$.155$ is the calculated [[loss]] for the model through categorical cross entropy.