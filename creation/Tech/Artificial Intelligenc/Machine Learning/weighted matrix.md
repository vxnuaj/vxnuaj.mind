The weighted matrix of a neuron, through it's individual [[weights]], $W_i$, allows for the calculation of the [[weighted sum]] through multiplying $W_i$ with the respective input value

In the matrix, the number of rows represents the number of neurons at the current layer, the [[hidden size]], and the columns represents the size of input neurons / data points.

For example, a weighted matrix with dimensionality of 3x3, 

$[W_1,W_2,W_3]$
$[W_4,W_5,W_6]$
$[W_7,W_8,W_9]$

indicates that the current layer has a size of 3, taking in data from 3 different neurons.

This weighted matrix is multiplied by by each individual value from the 3 different input neurons. A [[bias]] parameter is added to the result in order to calculate the [[weighted sum]].
