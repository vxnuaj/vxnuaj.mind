The weighted matrix of a neuron, through it's individual [[weights]] allows for the calculation of the weighted sum by multiplying it's individual weights per input token.

In the matrix, the number of rows represents the number of neurons at the current layer, the [[hidden size]], and the columns represents the size of input neurons / data points.

For example, a weighted matrix with dimensionality of 3x3, 

$[w_1,w_2,w_3]$
$[w_4,w_5,w_6]$
$[w_7,w_8,w_9]$

indicates that the current layer has a size of 3, taking in data from 3 different neurons.

This weighted matrix is multiplied by by each individual value from the 3 different input neurons. A [[bias]] parameter is added to the result in order to calculate the weighted sum

$x * w$
