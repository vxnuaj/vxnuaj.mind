---
Last Updated: 02-26-2024 | 7:19 PM
---
To evaluate multiclass problems through a variation of the [[accuracy]] equation, you can do so through:

$Multiclass Accuracy (y_{i},z_i)=\frac{1}{n}\sum_{i=1}^{N}[[y_i==z_i]]$

where
- $N$ is the total amount of samples fed into a model
- $i$ is the $ith$ sample fed into the model
- $y_i$ is the true class label for the sample $i$
- $z_i$ is the predicted class label for instance $i$
- $[[ ]]$ is the [Iverson bracket](https://en.wikipedia.org/wiki/Iverson_bracket), returning $1$ if $y_{1}==z_i$ and 0 otherwise

