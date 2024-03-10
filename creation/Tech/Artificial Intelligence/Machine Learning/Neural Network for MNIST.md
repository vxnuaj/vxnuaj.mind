---
created: 2024-02-19T10:40
Last Updated: 03-09-2024 | 10:17 PM
---
We will be building a neural network to classify digits from the MNIST dataset.

MNIST training set consists of handwritten digits with a dimensionality of 28 by 28  pixels, amounting to a total of 784 pixel values. Each numerical pixel value ranges from 0 to 255, with 0 representing black and 255 representing white.

This is what a set of handwritten MNIST digits look like:

![[Screenshot 2024-03-07 at 2.09.02 PM.png|300]]

Our dataset holds a total of 60,000 digits for training our model and 10,000 digits for testing our model.

Our network will be able to identify the numerical value of each handwritten digit.

It will consist of three layers.

The **Input Layer** will take in `784` datapoints, representing a pixel value from each 28 by 28 image of a handwritten MNIST digit

The **Hidden Layer** will take in the fed-forward `784` datapoints into it's `32` neurons and output `32` values.

The **Output Layer** will take in the `32` output values from the hidden layer into it's `10` neurons and output `10` activation values, of which the neuron with the highest activation value will correspond to the networks final digit prediction

We import the dependencies for our program

```
import numpy as np
import pandas as pd
import pickle
```

NumPy is for data manipulation & linear algebra.
Pandas is for reading the dataset from CSV format.
Pickle will allow us to save the parameters of our trained model as a byte stream.

First things first, we define the functions that will allow us to save our model once it's trained and load our trained model for usage.

```
def save_model(w1, b1, w2, b2, filename):
	with open(filename, 'wb') as f:
		pickle.dump((w1, b1, w2, b2), f)

def load_model(filename):
	with open(filename, 'rb') as f:
		return pickle.load(f)
```

The `save_model` function will take in the parameters of our neural network `w1, b1, w2, b2` and the `filename` we wish to assign to the file `pickle.dump` will create (which will be defined as `.pkl` later). With the `open()` function, we open a defined file in the mode `wb`, which indicates `w`riting to the file in `b`inary, to variable `f`.

Then we use `pickle.dump()` to write our parameters, `w1, b1, w2, b2` to our opened file, to save it for future use.

The `load_model` function does a similar job, but instead of saving the model parameters, it loads our parameters into our program, by reading the sequence of bytes (`rb` mode) from the file, onto variable `f`. `pickle.load()` is then used to get the parameters from the file to load onto our program

Given that the MNIST dataset within our directory in csv format, we use the `pd.read_csv`function to access data from the csv file and upload it as a pandas DataFrame.

```
train_data = pd.read_csv("../MNIST CSV archive/mnist_train.csv")
train_data.head(5) ## Optional if in Jupyter Notebook.
```

This is assigned to `train_data`, & to confirm that it worked, you can print the first 5 rows of the dataset using the `pd.head(5)` function if you're in a Jupyter notebook

You should get something similar, if not the same, to this

![[MNISTReadCSVResults.png]]

Each row represents each digit in the dataset while each column corresponds to each pixel value for the corresponding digit / row.

Now that we've verified that our dataset is loaded into our program, we can safely begin pre-processing our dataset to prepare it for our model.

```
train_data = np.array(train_data)
m, n = train_data.shape
np.random.shuffle(train_data)
```

We turn our DataFrame into a NumPy array which will allow us to perform mathematical operations on our dataset.

The rows of our NumPy array, `train_data`, is assigned to variable `m` and the columns are assigned to variable `n`. 

The rows or `m` is equal to the total number of samples (digits) in our dataset
The columns or `n` is equal to the total number of pixel values in our dataset.

`m` will allow us to 
- Average the gradient of the loss with respect to our weights and biases over all samples, `m` which will allow for our model to learn as it's fed more digits.
- Compute the average loss using [[categorical cross entropy]] per samples `m`.

`n` will allow us to
- Split our training data and our labels from our NumPy array.

Finally, we use `np.random.shuffle()` to randomize the order of our training data to prevent overfitting when we begin to train our model.

To further pre-process our data,

```
train_data = train_data[0:m].T
Y_train = train_data[0]
X_train = train_data[1:n]
X_train = X_train / 255
```

we take our `train_data` and transpose it over all samples `m`.

Our labels, now located in the first row of our dataset (given that we transposed `train_data`), denoted by index `0`, will be extracted accessed by indexing `train_data` and row `[0]`. We assign the labels to `Y_train`.

The pixel values of each individual digit in the dataset will be accessed by indexing `train_data` from row `1` to `n` samples. We assign the training data to `X_train`. 

Afterward, we normalize our datapoints to between the range 0-1 by diving `X_train` by `255`. This will allow for our model to more effectively compute a prediction by applying its parameters through a [[weighted sum]] & [[activation function]] and then learn faster through [[backpropagation]] and [[gradient descent]]. 

If in a Jupyter notebook, you can check the total size of our labels, `Y_train`, by running,

```
Y_train.size
```

You should get `60000` as the output, representing the total number of labels that correspond to our `60000` digit samples. 

You can also run,

```
X_train.shape
```

to get the shape of our training data. You should get `784, 60000` as the output, 784 being the number of pixel values per each 28 by 28 digit in the `60000` total digit samples

Now, we can begin to define our initial functions that make up our model.

```
def init_params():
	w1 = np.random.rand(32, 784) - .5
	b1 = np.zeros((32,1)) - .5
	w2 = np.random.rand(10, 32) - .5
	b2 = np.zeros((10,1)) - .5
	return w1, b1, w2, b2
```

We will initialize our first weights matrix, `w1` by creating a random array of floats with a dimension of `32, 784`. The `32` rows represent the number of output neurons in our first hidden layer and the `784` columns are the number of input pixel values into our first hidden layer.

Our first [[bias]] matrix, `b1` will be initialized using `np.zeros`, creating a matrix of 0s, with a dimensionality of `32` rows and `1` column. The `32` represents each input neuron in our first hidden layer. We only use 1 column as we only need `32` bias parameters for each neuron, nothing more.

Our second [[weight]] matrix `w2` will be a dimensionality of `10,32`, initialized using `np.random.rand`, to again create a random array of floats. `10` represents the `10` neurons in the output layer and `32` represents the `32` inputs from the `32` neurons in the hidden layer to the output layer.

The second [[bias]] matrix, `b2` will be another matrix of `np.zeros`, with a dimensionality of `10` rows and `1` column. `10` being the total number of bias values we only. `1` column as we only need a total of `10` bias values.

Now, the activation functions we'll be using, [[ReLU]] and [[SoftMax]], will be initialized.

```
def relu(z): #ReLU
	return np.maximum(z,0) # np.maximum(z,0) is ReLU

def softmax(z): #Softmax
	A = np.exp(z)/ np.sum(np.exp(z))
	return A
```

Mathematically, [[ReLU]] is simply defined as $f(z) = max(0,z)$, where $z$ is the input weighted sum for a given neuron. In NumPy, this can simply be expressed as `np.maximum(z,0)`

Now [[SoftMax]], which will be used for our output layer to compute probabilities per output neuron, is a little more mathematically complex. 

[[SoftMax]] can be defined as $\hat y_{i}=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}$, where 
- $i$ is the $ith$ output of the final layer, dependent on the number of neurons in the final layer. 
- $j$ is the index for the total number of classes, $K$, in a dataset. 
  In our case, $K$ is $10$, as our model has $10$ classes ranging from 0-9

In NumPy, this function can be expressed as `np.exp(z)/np.sum(np.exp(z))`.

Now, we can begin to define our forward function, which will allow us to feed our data through our network, applying the initialized [[weight]] and [[bias]]es, to calculate the [[weighted sum]]s and the activation outputs.

```
def forward(w1, b1, w2, b2, X)
	z1 = w1.dot(X) + b1
	a1 = relu(z1)
	z2 = w2.dot(a1) + b2
	a2 = softmax(z2)
	return z1, a1, z2, a2
```

The first calculation within this function computes the [[weighted sum]] of the hidden layer in our network. 

Before we go further, lets talk about the [[weighted sum]].

The [[weighted sum]] is essentially the sum of the multiplication of each [[weight]] and input at a given neuron. Once this sum is calculated, a bias parameter is then added for a final value.

The equation is defined as, $z = \sum_{i=1}^{N}w_{ij}·x_{ij}+ b_i$, where,

- $N$ is the total number of neurons in a current layer
- $i$ is the index of $ith$ neuron in the current layer
- $j$ is the index of $jth$ neuron from a previous layer connected to $ith$ neuron
- $w_{ij}$ is a weight at the connection between $jth$ and $ith$ neuron
- $x_{ij}$ is a given input at $ith$ neuron from $jth$ neuron at the previous layer
- $b_i$ is the bias for $ith$ neuron

In our code, `w1` represents the initialized matrix of weights with a dimensionality of `32, 784`. Rather than calculating the sum of each individual weight $w_{ij}$ multiplied by an input, $x_{ij}$, through `np.sum` we can take the dot product of the matrices `w1` and `X`, which already then calculates a summation of the multiplication.

The product of this operation will result in a final matrix, `z1` which contains the weighted sums of the hidden layer at each of the 32 neurons.

Now, we apply the [[ReLU]] activation function, $f(z) = max(0,z)$, through our defined function `relu(z)`, by taking in `z1` as its input. The resulting output is the activation matrix which holds an activation value each of the `32` neurons.

To calculate `z2`, the [[logits]] / [[weighted sum]] of the output layer, we do the same [[weighted sum]] calculation where we multiply by the input, this time `a1` representing the output of the hidden layer, by the weighted matrix `w2`. 

Our output, matrix `z2` holds the weighted sums of each of the `10` neurons at the output layer. We apply `z2` to our defined [[SoftMax]] activation function, $\hat y_{i}=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}$, to calculate the final probabilities for each of our 10 classes in our dataset (digits 0 to 9).

Up to here, we've fully defined the feed forward mechanism of our neural network. Now, we can begin to initialize some of the functions that will begin to make up the backpropagation for training our network.

First, we begin to define the function that takes the derivative of the [[ReLU]] activation function. This will help us calculate the gradients (aka derivatives) of our loss function (which will be [[categorical cross entropy]]), with respect to a given parameter.

```
def relu_deriv(z):
	return z > 0
```

Given that [[ReLU]], when the input is greater than 0, is a linear function with a slope of 1, the derivative of [[ReLU]] when the value is greater than 0 will always return 1.

When ReLU is ≤ 0, it always returns 0 hence the derivative will always be 0.

Therefore, we can simply return `z > 0` as this gives us a boolean matrix that assesses whether or not `z` is greater than 0. If `z` is greater than 0, it will return True, representing 1, and if not, it will return False, representing 0. 

Now the function defining the [[one-hot encoding]] vector of a corresponding label for an input digit will be initialized.

A [[one-hot encoding]] is a way of expressing the labels for a dataset in a binary format.

It transforms the labels of a dataset into vectors of binary 1s and 0s to allow for a model to more easily interpret & then classify pieces of data.

These vectors have an equal length to the number of classes in a dataset.

For each one-hot encoded vector, each binary element will be a 0 except for the element that identifies a specific class, which will be set to 1.

For example, if I had a dataset of 3 classes ranging from digits 0 to 2, the one hot encoded vector for each would be:

- 0 | $[1, 0, 0]$
- 1 | $[0,1,0]$
- 2 | $[0,0,1]$

```
def one_hot(Y):
	one_hot_Y = np.zeros((Y.size, Y.max() + 1))
	one_hot_Y[np.arange(Y.size), Y] = 1
	one_hot_Y = one_hot_Y.T
	return one_hot_Y
```

To initialize our matrix for our [[one-hot encoding]], we create a new matrix of 0s through `np.zeros`, of a dimensionality of `Y.size, Y.max() + 1`.

> *Remember, Y represents the total labels of our dataset, 1 label per digit sample.*

Essentially, we initialize a matrix with each row representing each digit in our dataset (per `Y.size`) while the columns represent the total classes of our dataset which amounts to 10.

> *`Y.max()` calculates the maximum value in our labels which is 9 so we `+ 1` to get a total of 10 for the length of each row*

The dimensionality for our [[one-hot encoding]] matrix should end up as `60000, 10`.

Now, we can index our matrix to assign a value 0 to 1, which will indicate a corresponding label from our dataset.

Using `np.arange` we create an matrix of length `Y.size`, which in our case, will be `60000`, as our training set holds `60000` digits. This matrix holds values ranging from 0 to `60000`, with a step size of 1

Like this: $[0,1, ... 59999, 60000]$

So we index our `one_hot_y` matrix at `np.arange(Y.size), Y`. 

Essentially, given that the matrix defined by `np.arange(Y.size)` has `60000` rows, for each `60000` rows, we index a 0 at value `Y` and then set it equal to 1.

As an example, If the first row in `one_hot_Y` was indexed by a `Y` value of 3 and the second row was indexed by a value of 6, the first two rows of `one_hot_Y` would look like:

$[0,0,0,1,0,0,0,0,0,0]$
$[0,0,0,0,0,0,1,0,0,0]$

Then, we transpose the `one_hot_Y` matrix. This will allow us to match the dimensionality of this matrix with `a2`. Later on, during [[backpropagation]], this will allow us to calculate the derivative of the loss function with the weighted sum `z2` (you'll see later!).

Now, we can define our [[loss function]]. 

We will be using [[categorical cross entropy]], a loss function designed for multiclass classification problems just like the one we're dealing with right now with the MNIST dataset!

It calculates the loss between the predicted probability and a true value by taking the logarithm of the predicted probability and multiplying it by a true value. 

It's calculated by $y_i·log(\hat{y_i})$, where $y_i$ is the true value and $\hat{y}_i$ is the predicted probability for $ith$ class of a dataset.

In our network, we'll want to calculate the total loss of our network over all neurons / classes. So what we do is apply

cool ;)
---

The initial parameters, [[weight]] and [[bias]], for our function alongside the number of neurons are defined.

Our network will have an input layer of `784` Neurons, as each image within the MNIST dataset consists of `784` pixels, in a 28 by 28 pixel grid.

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

Ultimately `dw2` / $\frac{∂C_o}{∂w_2}$ defines a matrix of the gradients of the [[loss function]] with respect to [[weight]] $w_2$ for each and every connection between each neuron $m$ and $n$

The output `dw2` is then a matrix, of dimensions $(n,m)$ where $n$ is the number of neurons in the output layer and $m$ is the number of neurons in the input layer.

To compute `db2`, the derivative of the loss function with respect to the bias in a neuron $n$, we can take the sum of `dz2` on the first axis. Meaning, we sum up the gradients of the loss function with respect to the [[weighted sum]] of the 10 neurons, $n$.

Mathematically, this can be defined as,

$\frac{∂C_o}{∂b_{2}}=(\frac{∂C_0}{∂z_2})(\frac{∂z_2}{∂b_2})$

Given that the bias, `b2` isn't part of a multiplication with the activation function, $a_2$, we don't need to take the gradient of the loss w.r.t `a2`, we can instead directly calculate $(\frac{∂C_0}{∂z_2})$.

Since the derivative of a variable with respect to itself is 1,

$\frac{∂C_o}{∂b_{2}}=(\frac{∂C_0}{∂z_2})1$

This is summed up over all 10 neurons in the output layer,

$\frac{∂C_o}{∂b_{2}}=\sum_{i = 1}^{10}(\frac{∂C_0}{∂z_2})$

The same bias is defined per layer, for each and every neuron, therefore the summation of gradient of the loss w.r.t to `z2` over every neuron, is used to calculate `db2`.

To calculate the gradient of the loss with respect to the weighted sum of the first layer, `dz1`, we can multiply the gradient of activation function output, ReLU(z1) / `a1`, with the dot product of `w2.T` and `dz2`.

Mathematically, `dz1 = relu_deriv(z1) * (np.dot(w2.T, dz2))` looks like this:

$\frac{∂C_0}{∂z_{1}} = (\frac{∂a_1}{∂z_{1}})(\frac{∂z_2}{∂a_1})(\frac{∂C_0}{∂z_2})$.

The gradient of the loss w.r.t to the weight in the hidden layer. `dw1` can be calculated by taking the dot product of `dz1` and the transposed inputs, `X.T`. Mathematically it looks like this:

$\frac{∂C_o}{∂w_{1}}= (\frac{∂C_0}{∂a_1})(\frac{∂a_1}{∂z_1})(\frac{∂z_1}{∂w_1})$

Now `db1`, similar to `db2` can be calculated as the sum of `dz1`,

$\frac{∂C_o}{∂b_1}=\sum_{i=1}^{32}\frac{∂C_o}{∂z_1}$

over a total of $32$ neurons in the hidden layer.