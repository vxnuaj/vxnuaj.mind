Categorical Cross Entropy Loss is a [[loss function]] which calculates the [[loss]] of a model's output.

It's specifically designed for multi-class classification tasks where each input belongs exactly to one class.

The equation is defined as

$CE = -\sum_{i=1}^{N}y_{i}*log(\hat{y_i})$

where,
- $\hat{y}_i$ is the prediction of a model
- $y_i$ represents the actual value of the input
- Index $i$ denotes the $ith$ class of the dataset
- $N$ denotes the total number of classes in the dataset
- 

