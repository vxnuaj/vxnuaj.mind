---
created: 2024-02-19T10:40
Last Updated: 02-21-2024 | 12:53 noon
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
train_data = np.array(train_data)
Y_train = train_data[:,0] #Getting our training labels
X_train = train_data[:,1:] #Getting our training data
```

We convert our csv file into a NumPy Array and slice the NumPy array to split the training labels and training features.

`Y_train` are the labels of our training data.
`X_train` is the actual training data.

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
    z2 = np.dot(w2, a1) + b2.reshape(-1,1)
    a2 = softmax(z2)
	return z1, a1, z2, a2
```

Here, `z1`, the [[weighted sum]] of each neuron in the first layer, is computed by taking the dot product of [[weighted matrix]] `W1` and `X` which represents the original input of `784` pixels.

Afterward, `A1`, the final output(s) of the 1st layer, is calculated by applying our [[ReLU]] activation function onto the [[weighted sum]] `Z1`.

`z2`, the [[weighted sum]] of each neuron in the second layer is computed as the dot product of [[weighted matrix]], `W2`, and `A1`. 

We get our final output, `A2` by applying non-linearity to `z2` through the [[SoftMax]] [[activation function]].

Now the dataset will be taken and through [[one-hot encoding]], will be transformed into numerical data of vectors of 0s and 1s.

This is done in order to allow for our [[neural network]] to interpret the dataset.

```
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
```

Within the function `one_hot()`, we'll take in the training labels `Y_train`. 

We create NumPy array filled with 0s with a dimensionality of `(Y.size, Y.max() + 1)`, where `Y.size` is the total # of labels in our training set defining the total # of digits in our MNIST training set. `Y.max()`, defines the largest value of our training set `Y`, which in the case of MNIST database, is 9.

Now in the MNIST dataset, 0-9 accounts for `10` digits, if you count 0.
Therefore, we must `+1` to `Y.max()` in order to ensure that our array will account for all digits in the MNIST dataset.

Our NumPy array of 0s will have the dimensionality of `(60000, 10)`, as our dataset holds `60000` digits ranging from 0 to `9`.

Now, using `np.arange` we create an array of length `Y.size`, which in our case, will be `60000`, as our training set holds `60000` digits. It holds values ranging from 0 to 60000.

Like this: $[0,1, 2, 3, ... 59998, 59999, 60000]$

Our `one_hot_y` array of `np.zeros` will be indexed using `np.arange(Y.size)` and `Y`. 

Essentially, what we're doing is for every value $i$ in `np.arange(Y.size)`, we take $i$ and the label `Y`, index our `one_hot_Y` array, and assign that specific indice `0` to `1`.

The value `Y` is dependent on which image is fed into the network.
If we feed in an image that represents the digit 5, `Y` will be 5, representing the 5th indice of label array `Y`, $[0,1,2,3,4,|5|,6,7,8,9,]$

So if we ran this function for the first 5 digits in the MNIST dataset,

$[5]$
$[0]$
$[4]$
$[1]$
$[9]$

our `one_hot_y` encoded array would look something like this:

$[0,0,0,0,1,0,0,0,0,0]$
$[1,0,0,0,0,0,0,0,0,0]$
$[0,0,0,0,1,0,0,0,0,0]$
$[0,1,0,0,0,0,0,0,0,0]$
$[0,0,0,0,0,0,0,0,0,1]$

Then our `one_hot_y` array is transposed per `np.T`, and our function returns the `one_hot_y`.


### Next Steps
- [ ] Define Categorical Cross Entropy
- [ ] Define Back Propagation