---
{}
---
Dependences are imported for the program

```
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
```

NumPy is for data manipulation & linear algebra.
Pandas is for reading the Dataset.
PyPlot is for visualization of the dataset

```
train_data = pd.read_csv("MNIST CSV archive/mnist_train.csv")

print(train_data.head) ## Optional if in Jupyter Notebook.
```

Given that the MNIST dataset is within our directory in csv format, we use the `pd.read_csv` function to access data from the csv file and upload it as a pandas DataFrame.

This is assigned to `train_data`, & to confirm that it worked, we can print the first few rows of the dataset using the `.head` function.

You should get something similar, if not the same, to this

![[MNISTReadCSVResults.png]]


```
train_data = np.array(train_data) #Converting our CSV into a NumPy Array
train_labels = train_data[:,0] #Getting our training labels
train_features = train_data[:,1:] #Getting our training data
```

We convert our csv file into a NumPy Array and slice the NumPy array to split the training labels and training features.

```
def init_params():
    w1 = np.random.rand(32, 784) # W matrix 1 of 32 (neurons) x 784 (inputs)
    b1 = np.zeros(784) # bias 1 of size 784 to add per 784 weighted sums.
    w2 = np.random.rand(10, 32) # W matrix 2 of 10 (neurons) x 32 (inputs)
    b2 = np.zeroes(32) #bias 2 of size 32 to add per 32 weighted sums.
    return w1, b2, w2, b2
```

The initial parameters, [[weights]] and [[bias]], for our function alongside the number of neurons are defined.

Our network will have an input layer of `784` Neurons, as each image within the MNIST dataset consists of `784` pixels.

![[MNIST Sample.png| Images just like these]]

It will have a hidden layer of `32` neurons. So, hence `W1`, it represents the [[weighted matrix]] of the [[hidden layer]] taking in `784` input neurons and outputting `32` values from it's `32` neurons.

A [[bias]] vector, `b1` is created of a length of `784` to accommodate for each `784` neuron. Each [[bias]] will be added to the multiplication of [[weight]] $W_i$ and output $x_i$, where $i$ is the $ith$ neuron, as part of the [[weighted sum]] calculation.

The output layer will take in `32` values from the previous `32` neurons. It will consist of `10` neurons, outputting `10` final values. `W2` is defined according to those parameters.

A [[bias]] vector, `b2`, is created this time with a length of `32` to accommodate for the `32` neurons in the output layer.

We create the [[bias]] using `np.zeros` just as a means of initializing the model parameters. It will change as we train out network.

The same concept is done with the [[weight]]s, instead using `np.random.rand`

Prior to defining the forward function, we define our [[activation function]]s.
We'll be using [[ReLU]] and [[SoftMax]].

```
def relu(z):
    return np.maximum(z,0)

def softmax(z):
    return (np.exp(x)/np.exp(x).sum())
```

In NumPy, the ReLU activation function can be defined by using the np.maximum function, with parameters `z,0`.

The SoftMax function can be defined through the equation $\frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$, which is what `np.exp(x)/np.exp(x).sum()` defines

Now, the forward function can be defined.

```
def forward(w1, b1, w2, b2, X):
    z1 = np.dot(w1,X) + b1.reshape(-1,1) 
    a1 = relu(z1)
    z2 = np.dot(w2,X) + b2.reshape(-1,1)
    a2 = softmax(z1)
	return z1, a1, z2, a2
```

Here, $z_1$, the weighted sum of each neuron in the first layer, is computed by taking the dot product of matrix `W1` and `X` which represents the original input of `784` pixels.

