---
created: 2024-02-19T10:40
Last Updated: 02-24-2024 | 10:47 PM
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
    b1 = np.zeros(32) # bias 1 of size 32 to add per 32 2nd layer neurons
    w2 = np.random.rand(10, 32) # W matrix 2 of 10 (neurons) x 32 (inputs)
    b2 = np.zeros(10) #bias 2 of size 10 to add per final 10 output neurons
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

Afterwards, the [[loss function]] [[categorical cross entropy]] is defined. This [[loss function]] is made specifically for multi-class classification tasks where each input belongs exactly to a single class.

The equation of [[categorical cross entropy]] is defined as:

$CE = -\sum_{i=1}^{N}y_{i}·log(\hat{y_i})$

 - $\hat{y}_i$ is the predicted probability for $ith$ class in the dataset given by the model
- $y_i$ represents the one hot encoded vector for $ith$ class label
- Index $i$ denotes the index of the $ith$ class of a dataset
- $N$ denotes the total number of classes in the dataset / [[one-hot encoding]] vector.

Therefore, we implement it in our code as,

```
def cat_cross_entropy()
    CE = -np.sum(one_hot_Y * np.log(a2))
    return CE
```

We define the derivative of [[ReLU]], which for all numbers greater than 0, it's equal to 1 given that ReLU is linear when x > 0.

```
def relu_deriv(z)
	return (z>0).astype(int)
```

So the derivative can be simply returned as `z>0`, with `.astype(int)` to ensure that it returns as an integer value, not a boolean value.

Now, we can define our [[backpropagation]] function,

```
def back_prop(z1, a1, z2, a2, w2, X, Y ):
	one_hot_Y = one_hot(Y)
	dz2 = a2 - one_hot_Y
	dw2 = np.dot(dz2, a1.T)
	db2 = np.sum(dz2, axis = 1, keepdims=True)
	dz1 = relu_deriv(z1) * (np.dot(w2.T, dz2))
	dw1 = np.dot(dz1, X.T)
	db1 = np.sum(dz1, axis = 1, keepdims=True)
	return dw1, db1, dw2, db2
```

Through our [[one-hot encoding]] function, we encode data labels `Y`, & it set equal to `one_hot_Y`. 

The `one_hot_Y` encoded vector is used to calculate the derivative `dz2` by subtracting `a2` by the `one_hot_Y`.

To break down the math,

- $C_o$ is our loss function, [[categorical cross entropy]].
- $w_2$ / `w2` is the weight which connects our hidden layer to the output layer.
- $z_2$ / `z2` is the weighted sum from our hidden layer being fed as an input to the output layer through activation function $a_2$.
- $\hat{y}$ is the one hot encoded vector.
- $l$ is the $lth$ layer in range $0≤l≤2$. (If $j = l$, we are at the 1st hidden layer.)
- $m$ is each neuron in the output layer
- $n$ is each neuron in the input layer

Given that we're implementing the [[categorical cross entropy]], the derivative of `z2`, `dz2`, can be calculated by, `a2 - one_hot_y`.

$\frac{∂C_{o}}{∂z_2} = a_{2}- \hat{y}$ 

> *Check out the full derivation [here](https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)*

This indicates that a change to the gradient of the loss function, $C_o$, with respect to weighted sum, $z_2$, can be induced from a change in the activation output $a_2$.

Ultimately, the calculation results in `dz2` being a matrix that contains the gradient of the loss with respect to the weighted sum `dz2`, where each row represents individual training samples and each column is each individual neuron (10) in the output layer.

It has a dimensionality of $(samples, 10)$.

Then, `dw2` is the dot product of `dz2` and `a1.T`, essentially being the gradient of the loss function with respect to weight `w2`.

$\frac{∂C_o}{∂w_{2}}= (\frac{∂a_2}{∂z_2})(\frac{∂z_2}{∂w_2})$

The $\frac{∂z_2}{∂w_2}$, can simply be rewritten as the transposed $a_1$, as $a^{T}_{1}$,

$\frac{∂C_o}{∂w_{2}}= (\frac{∂a_2}{∂z_2})a^{T}_1$

Given that we know $\frac{∂a_2}{∂z_2}$ is equal to $a_{2}- \hat{y}$, we can rewrite our equation as,

$\frac{∂C_o}{∂w_{2}}= (a_2-\hat{y})a^{T}_1$

indicating that any change in $a_2$ will also ultimately affect the gradient of loss function $C_o$ with respect to $w_2$.

Given that `dz2` has a dimensionality of $(samples, 10)$ and the transposed `a1` has a dimensionality of $(samples, 32)$, we end up with the matrix `dw2` of a dimensionality of $(32, 10)$, where $32$ are the neurons in the hidden layer and $10$ are the neurons in the output layer. 

Ultimately `dw2` / $\frac{∂C_o}{∂w_2}$ defines a matrix of the gradients of the [[loss function]] with respect to [[weights]] $w_2$ for each and every connection between each neuron $m$ and $n$

The output `dw2` is then a matrix, of dimensions $(n,m)$ where $n$ is the number of neurons in the output layer and $m$ is the number of neurons in the input layer.

To compute `db2`, the derivative of the loss function with respect to the bias in a neuron $n$, we can take the sum of `dz2` on the first axis. Meaning, we sum up the gradients of the loss function with respect to the [[weighted sum]] of the 10 neurons, $n$.

Mathematically, this can be defined as,

$\frac{∂C_o}{∂b_{2}}=(\frac{∂C_0}{∂z_2})(\frac{∂z_2}{∂b_2})$

Since the derivative of a variable with respect to itself is 1,

$\frac{∂C_o}{∂b_{2}}=(\frac{∂C_0}{∂z_2})1$

This is summed up over all neurons $n$ in the output layer,

$\frac{∂C_o}{∂b_{2}}=\sum_{i = 1}^{n}(\frac{∂C_0}{∂z_2})$

The same bias is defined per layer, for each and every neuron, therefore the summation of gradient of the loss w.r.t to `z2` over every neuron, is used to calculate `db2`.