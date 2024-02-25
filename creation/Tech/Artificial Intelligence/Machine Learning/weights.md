---
Last Updated: 02-24-2024 | 8:39 PM
---
In a [[neural network]], weights represent the strength between 2 given neurons in 2 different layers.

These weights are multiplied by each input feature,

$x_{i} * w_{ji}$ (if input layer)
$a_{j}* w{ji}$ (if hidden or output layer)

to denote the activation of each neuron to a specific input which will serve useful for the [[weighted sum]] alongside a [[bias]] parameter.

The weights decide how a network interprets each input at a given training step, to ultimately be iteratively trained & adjusted through [[backpropagation]] for an increased level of accuracy.