---
created: 2024-02-19T10:40
Last Updated: 03-26-2024 | 10:31 PM
---
We will be building a neural network to classify digits from the MNIST dataset.

MNIST training set consists of handwritten digits with a dimensionality of 28 by 28 pixels, amounting to a total of 784 pixel values. Each numerical pixel value ranges from 0 to 255, with 0 representing black and 255 representing white.

This is what a set of handwritten MNIST digits look like:

![[Screenshot 2024-03-07 at 2.09.02 PM.png|500]]

Our dataset holds a total of 60,000 digits for training our model and 10,000 digits for testing our model.

Our network will be able to identify the numerical value of each handwritten digit.

It will consist of three layers.

The **Input Layer** will take in `784` datapoints, representing a pixel value from each 28 by 28 image of a handwritten MNIST digit

The **Hidden Layer** will take in the fed-forward `784` datapoints into it's `32` neurons and output `32` values.

The **Output Layer** will take in the `32` output values from the hidden layer into it's `10` neurons and output `10` activation values, of which the neuron with the highest activation value will correspond to the networks final digit prediction

#### Training the Model

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
train_data = pd.read_csv("MNIST CSV archive/mnist_train.csv")
## if in a jupyter notebook - ("../MNIST CSV archive/mnist_train.csv")
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

The rows of the NumPy array, `train_data`, is assigned to variable `m` and the columns are assigned to variable `n`. 

The rows or `m` is equal to the total number of samples (digits) in our dataset
The columns or `n` is equal to the total number of pixel values in our dataset.

`m` will allow us to 
- Average the gradient of the [[loss]] with respect to our weights and biases over all samples, `m` which will allow for our model to learn as it's fed more digits.
- Compute the average [[loss]] using [[categorical cross entropy]] per samples `m`.

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

Afterward, we normalize our datapoints to between the range 0-1 by dividing `X_train` by `255`. This will allow for our model to more effectively compute a prediction by applying its parameters through a [[weighted sum]] & an [[activation function]] to then learn faster through [[backpropagation]] and [[gradient descent]]. 

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

Our second [[weight]] matrix, `w2`, will be a dimensionality of `10,32`, initialized using `np.random.rand`, to again create a random array of floats. `10` represents the `10` neurons in the output layer and `32` represents the `32` inputs from the `32` neurons in the hidden layer to the output layer.

The second [[bias]] matrix, `b2` will be another matrix of `np.zeros`, with a dimensionality of `10` rows and `1` column. `10` being the total number of bias values we only. `1` column as we only need a total of `10` bias values.

Now, the activation functions we'll be using, [[ReLU]] and [[SoftMax]], will be initialized.

```
def relu(z): #ReLU
	return np.maximum(z,0) # np.maximum(z,0) is ReLU

def softmax(z): #Softmax
    return np.exp(z)/ sum(np.exp(z))
```

Mathematically, [[ReLU]] is simply defined as $f(z) = max(0,z)$ where $z$ is the input weighted sum for a given neuron. In NumPy, this can simply be expressed as `np.maximum(z,0)`

Now [[SoftMax]], which will be used for our output layer to compute probabilities per output neuron, is a little more mathematically complex. 

[[SoftMax]] can be defined as $\hat y_{i}=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}$ where 
- $i$ is the $ith$ output of the final layer, dependent on the number of neurons in the final layer. -
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

Then, we apply the [[ReLU]] activation function, $f(z) = max(0,z)$, through our defined function `relu(z)`, by taking in `z1` as its input. The resulting output is the activation matrix which holds an activation value each of the `32` neurons.

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

```
def cat_cross_entropy(one_hot_Y, a2 ):
	CCE = -np.sum(one_hot_Y * np.log(a2)) * 1/m
	return CCE
```

It calculates the [[loss]] between the predicted probability and a true value by taking the logarithm of the predicted probability and multiplying it by a true value. 

It's calculated by $y_k·log(\hat{y_k})$, where $y_k$ is the true value and $\hat{y}_k$ is the predicted probability for $kth$ class of a dataset.

In our network, we'll want to calculate the total loss of our network over all neurons / classes. So what we do is apply a summation over all $i$ classes in our dataset.

The final equation will be $CCE = -\sum_{k=1}^{K}y_k·log(\hat{y}_k)$, where $K$ is the total classes in our dataset.

This is represented by `-np.sum(one_hot_Y * np.log(a2))` in NumPy.

We divide this over `m` in order to get the average loss over the total samples in our dataset.

**Finally, we can define our function for [[backpropagation]].** 

For a little more context, backpropagation is the process where we take the gradients (derivatives) of a loss function with respect to a specific parameter (a weight or bias) or output in order to minimize the value of the gradient through an optimization algorithm such as [[gradient descent]].

Minimizing the gradient of the loss function to near 0 indicates that the loss of our neural network is near 0. This is exactly what we want for an increased accuracy.

In context of [[gradient descent]], let's say our loss function is in the form of a parabola.

(for an intuitive understanding)

We begin with an initial value, at the left of the parabola.

![[gradientdescent1.png|500]]
The goal here is to iteratively minimize the tangent of the value.

After the first iteration of [[gradient descent]], our value and parabola might look like this:
![[gradientdescent2.png | 500]]
The tangent of the value has decreased. As you might've deduced, we ultimately want the value to reach the trough of the parabola where the tangent is 0.

![[gradientdescent3.png|It's rare to see an optimal value, there will often always be a loss|500]]

This process is repeated iteratively for all $i$ neurons over all $l$ layers through the process of [[backpropagation]].

> *It involves some pretty dense mathematics, which we'll go into*

```
def back_prop(z1, a1, z2, a2, w1, w2, X, Y):
	one_hot_Y = one_hot(Y)
	dz2 = a2 - one_hot_Y
	dw2 = dz2.dot(a1.T) * 1/m
	db2 = np.sum(dz2) * 1/m
	dz1 = relu_deriv(z1) * w2.T.dot(dz2)
	dw1 = dz1.dot(X.T) * 1/m
	db1 = np.sum(dz1) * 1/m
	return dw1, db1, dw2, db2
```

We run our `one_hot()` function to apply [[one-hot encoding]] to our labels.

Now we calculate the derivative of the loss w.r.t to the weighted sum of the output layer by subtracting `a2 - one_hot_Y`.

Mathematically, the full equation (using chain rule) looks like this:

$\frac{∂C}{∂z_{2}}= (\frac{∂C}{∂a_2})(\frac{∂a_2}{∂z_2})$

This can be simplified as $\frac{∂C}{∂z_{2}}=a_{2}-\hat{y_k}$, where
- $C$ is the loss function
- $z_2$ is the weighted sum of the output layer
- $a_2$ is the activation output of the output layer
- $\hat{y_k}$ is the [[one-hot encoding]] of a label corresponding to the $kth$ class of our dataset

> *Check out the full derivation [here](https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/)*

Now that we've calculating `dz2`, we can calculate the derivative of the loss function with respect to our weights at the output layer, `w2`, through,

$\frac{∂C}{∂w_{2}}=(\frac{∂C}{∂z_2})(\frac{∂z_2}{∂w_2})$.

Note, we don't explicitly calculate $\frac{∂C}{∂a_2}$ as it's already done in the previous calculation of $\frac{∂C}{∂z_2}$.

This effectively outputs the [[loss]] for each weight at the output layer which will be extremely useful for the [[gradient descent]].

We can define this as `dz2.dot(a1.T)`where,

- `dz1` is $\frac{∂C}{∂z_2}$
- `a1.T` is $\frac{∂z_2}{∂w_2}$

Here's why $\frac{∂z_2}{∂w_2}$ is equivalent to the transposition of `a1`.

We calculate $z_2$ through $w_{2}·a_{1}+b_1$

To find the $\frac{∂z_2}{∂w_2}$, we're essentially taking the derivative of the matrix multiplication of $w_2$ and $a_1$. Whenever we take the derivative of a matrix multiplication with respect to one of the operands, we get transposition of the other operand which in this case is $a_1$.

Why the transposition?

In the original matrix multiplication, $z_{2}=w_{2}·a_1$, we're essentially multiplying the rows of $w_2$ by the columns of $a_1$. 

To keep the calculation of the derivative uniform and consistent with the original matrix multiplication, when we take the derivative of a matrix multiplication w.r.t an operand, we must transpose the remaining operand, which in this case is $a_1$

Now, we average the result from `dz2.dot(a1.T)` over the total `60000` samples by dividing over `m`. This is done in order to get the average loss over all samples.

Now, to calculate `db2`, we take a summation of `dz2`.

`db2 = np.sum(dz2) * 1/m`

We want to do so as we'll be using the total gradients of the loss w.r.t `dz2` for all 10 output neurons as our medium for error correcting `b2`.

Mathematically, this is defined as $\frac{∂C}{∂b_{2}}=(\frac{∂C}{∂z_2})(\frac{∂z_2}{∂b_2})$, but can be simplified to 

$\frac{∂C}{∂b_{2}}=(\frac{∂C}{∂z_2})1=(\frac{∂C}{∂z_2})$.

But we want to take the sum of `dz2` over all 10 output neurons so we can define this as,

$\frac{∂C_o}{∂b_{2}}=\sum_{i = 1}^{10}(\frac{∂C}{∂z_{2i}})$

where $i$ is the $ith$ neuron.

Again, this is averaged over all `60000` samples `m`.

Now, to calculate `dz1`, the gradient of the loss w.r.t to the weighted sum of the hidden layer, we take the derivative of [[ReLU]] and multiply it by the dot product of `w2.T` and `dz2`.

`dz1 = relu_deriv(z) * w2.T.dot(dz2)`

Mathematically, this looks like $\frac{∂C}{∂z_{1}}= (\frac{∂C}{∂z_2})(\frac{∂z_2}{∂a_1})(\frac{∂a_1}{∂z_1})$ where, 

- $(\frac{∂a_1}{∂z_1})$ is `relu_deriv(z)`
- $(\frac{∂z_2}{∂a_1})$ is `w2.T`
- $(\frac{∂C}{∂z_2})$ is `dz2`

We can now use our definition of `dz1` to calculate `dw1`, the gradient of the loss with respect to the weights in the hidden layer.

`dw1 = np.dot(dz1, X.T)` can be expressed as,

$\frac{∂C}{∂w_{1}} = ({\frac{∂C}{∂z_1}})(\frac{∂z_1}{∂w_1})$.

Here, the transposition of `X`, `X.T` is essentially $\frac{∂z_1}{∂w_1}$. 

Since we must compute the matrix multiplication of $w_1$ and the input matrix $X$ to get the weighted sum $z_1$, taking the derivative of $z_1$ w.r.t to one of the operands, in this case $w_1$, will result in the transposition of the remaining operand, $X$, ultimately yielding `X.T`.

Once again, we average this by dividing over our 60000 samples, `m`

Finally, to calculate the gradient of the loss w.r.t to our final parameter `b1`, we can take the sum of `dz1`.

`db1 = np.sum(dz1)`

Just as earlier, we want to take the sum of the weighted outputs in our hidden layer as we'll be using this metric for error correcting `b1`.

We divide this over our 60000 samples, `m` to get the average gradient over all samples.

Once defining the function for [[backpropagation]], now we can define the function which represents our update rule. 

It will directly update our parameters based on the prior calculations of the gradient.

```
def update_params(w1, b1, w2, b2, alpha)
	w1 = w1 - alpha * dw1
	b1 = b1 - alpha * db1
	w2 = w2 - alpha * dw2
	b2 = b2 - alpha * db2
	return w1, b1, w2, b2
```

This process defines the structure of the [[gradient descent]] algorithm, through the update rule of, $\uptheta = \uptheta - \upalpha*\frac{∂C}{∂\uptheta}$.

For each parameter, `w1`, `b1`, `w2`, `b2`, we apply this update rule to ultimately update our parameters towards an optimal value through [[gradient descent]]

Now, we can define the function which will get the final prediction of our model from the estimated probabilities.

```
def get_pred(a2)
	pred = np.argmax(a2, axis = 0)
	return pred
```

Through the function, `np.argmax()`, we get the maximum value of vector `a2`, to get the highest probability predicted over all classes in our dataset.

Now we define the function which will dynamically compute the accuracy of our model and print 

```
def accuracy(predictions, Y)
	acc = np.sum(predictions == Y) / Y.size
	return acc
```

We take the summation of total predictions that is equal to a label `Y`. This essentially acts as a mechanism, checking if a prediction is correct and then adding it to a summation. 

We divide this over the total size of labels `Y` to get the final accuracy value, `acc`.

Finally, we can define the final function which will ultimately perform [[gradient descent]], allowing for us to iteratively train our neural network for an increased accuracy.

```
def gradient_descent(X, Y, alpha, iterations):
	model_filename = 'models/mnistnn.pkl'
	try:
		w1, b1, w2, b2 = load_model(model_filename)
		print("Loaded model from:", model_filename)
	except FileNotFoundError:
		print("Model not found. Initializing new model!")
		w1, b1, w2, b2 = init_params()
		
	for i in range(iterations):
		z1, a1, z2, a2 = forward(w1, b1, w2, b2, X)
		dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w1, w2, X, Y)
		w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha )
		if i % 10 == 0:
			loss = cat_cross_entropy(one_hot(Y), a2)
			acc = accuracy(predictions, Y)
			print("Iteration: ", i)
			print("Accuracy:", acc)
			print("Loss:", loss)
			print(f"Predictions: {predictions} | True Vals: {Y}")
		
	return w1, b1, w2, b2 
```

`iterations` is the total number of times we feed our dataset into our network for training and `alpha` is the learning rate which defines how big of a step our model takes when training through [[gradient descent]].

You might recall alpha $(\upalpha)$ being used in the update rule for [[gradient descent]] 

$\uptheta = \uptheta - \upalpha*\frac{∂C}{∂\uptheta}$

Mathematically, that's how alpha is used in a deep learning model.

Now first off, we attempt to load a previously trained model. If one has been trained previously and exists in our directory, we print `Loaded model from: mnistnn.pkl`.

Otherwise, we print `Model not found. Initializing new model!` and initialize our parameters, `w1, b1, w2, b2` through our `init_params()` function.

Now, to define the main structure of this function, we want our `forward()`, `back_prop()`, and `update_params()` functions to be executed `i` times, according to the range in the total `iterations` that will be defined once the `gradient_descent` function is ran.

Now for every $i$ in the range, `iterations`, when the remainder of $i$ divided by 10 is equal to 0, we run:

- `cat_cross_entropy()` to get the average loss up to $ith$ iteration
-  `get_pred()` to get the prediction of our model to be used in `accuracy()`
- `accuracy()` to get the average accuracy up to $ith$ iteration

Finally, we print the current iteration, [[loss]], accuracy, and predictions / true values.

This is only printed when the remainder of $i$ is 0 as this allows us to visualize the training of our model per every 10th iteration.

Now finally, our model will be trained by running the `gradient_descent()` function.

```
w1, b1, w2, b2 = gradient_descent(X_train, Y_train, .1, 500)
save_model(w1, b1, w2, b2, 'models/mnistnn.pkl')
```

We define the learning rate at `.1` and the total iterations as `500`.
This effectively trains our model over `500` iterations through our dataset.

The model is then saved onto `mnistnn.pkl` using the `save_model()` function. 
This will allow us to load and continue training the model past `500` iterations if needed.

Here are my outputs / metrics per every 100th iteration when I first trained the model!

```
Iteration:  0
Accuracy: 0.09158333333333334
Loss: 3.918500600350083
[1 7 0 ... 9 0 1] [3 2 9 ... 9 5 2]

Iteration:  100
Accuracy: 0.7534333333333333
Loss: 0.7623425461001313
[1 2 9 ... 9 9 2] [3 2 9 ... 9 5 2]

Iteration:  200
Accuracy: 0.8235
Loss: 0.5627834257539484
[3 2 9 ... 9 5 2] [3 2 9 ... 9 5 2]

Iteration:  300
Accuracy: 0.8519
Loss: 0.48099627297439296
[3 2 9 ... 9 5 2] [3 2 9 ... 9 5 2]

Iteration:  400
Accuracy: 0.8686833333333334
Loss: 0.4330374059050395
[3 2 9 ... 9 5 2] [3 2 9 ... 9 5 2]

Iteration:  490
Accuracy: 0.8785
Loss: 0.40271510565759355
[3 2 9 ... 9 5 2] [3 2 9 ... 9 5 2]
```

By the final iteration, we achieve an accuracy of .88%.

#### Testing the Model

To see the model in action, we can simply run a single sample of the MNIST dataset through the network.

I'll be doing so in a new `.py` file.

First, just like always, import dependence!

```
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mnistmodel import load_model, get_pred
```

Per usual, just as before, we import NumPy and Pandas. Though this time, we import the `pyplot` module from matplotlib to allow for us to visualize the output of our network.

We also import the `load_model` and the `get_pred` functions from the file that holds the code for our model, `mnistmodel`.

We'll be using the `load_model` to load our trained model and `get_pred` to get the prediction from our model for visualization and testing.

Now, we can load our MNIST test data and pre-process it.

```
test_data = pd.read_csv(MNIST CSV archive/mnist_test.csv)
test_data = np.array(test_data)
m, n = test_data.shape # 10000, 784

test_data = [0:m].T
Y_test = test_data[0]
X_test = test_data[1:n]
X_test = X_test / 255
```

Just as before we load the `.csv` file and turn it into a NumPy array. We set `m` to be the rows of the dataset, 10000, and `n` to be the columns of the dataset, 784.

We then transpose our `test_data` array / matrix over all 10000 columns `m`.
Now, each column of our dataset represents each digit rather than each row.

Then we take the first column at index `0` of our `test_data`, which represents the labels of each digit sample, and assign it to `Y_test`.

The rest of the columns in `test_data`, from `1:n` are assigned to `X_test` which is then normalized to a decimal between 0 and 1 through dividing over `255`.

Now, we can load our model.

```
nn = 'models/mnistnn.pkl'
w1, b1, w2, b2 = load_model(nn)
```

We define our file path as `nn` and input it into the `load_model()` function which returns the trained parameters, `w1, b1, w2, b2`.

Now we can define the function that will take our neural network to output a prediction.

```
def make_pred(w1, b1, w2, b2, X, index):
	_, _, _, a2 = forward(w1, b1, w2, b2, X)
	prediction = get_pred(a2)
	prediction = prediction[index]
	true_digit = Y_test[index]
	print(f'Prediction: {prediction}')
	print(f"True Digit: {true_digit}")
	digit = X_test[:, index]
	digit = digit.reshape((28, 28)) * 255
	plt.gray()
	plt.imshow(digit)
	plt.show()
	return
```

We input our parameters alongside our test data into the `forward()` function to get the activation output `a2`.

The `get_pred()` function is used to get the `prediction` of the highest probability. 

This `prediction` is indexed using the `i` value, which will be input in to the `make_pred()` function when it is called.

This essentially gets the prediction for the specific digit at index $i$ that we're feeding to our network.

In order to test this predicted value against the true value, we can index our labels, `Y_test`, with the index, `i`.

This gets the true label, `true_digit` for our input digit sample.

Both, the `prediction` and the `true_digit` are printed onto the terminal.

We're also going to add the ability to visualize the specific digit at index $i$ by using the `pyplot` module.

We index our test data, `X_test` at all rows and column index, `i`, to get the specific pixel values for the digit at index `i`.

The shape of this sliced array is `784, 1` where `784` is the total pixel values for a digit.

To process this array for visualization we can simply reshape this array into a `28, 28` array. Once reshaping, we multiply by `255` in order to convert the normalized pixel values back to their original values.

Now, we generate the `pyplot` of our digit using `plt.gray(), plt.imshow(), and plt.show()`.

Let's test our model on the first value in our test set, at index `0`. 

This happens to be the digit 7.

```
prediction = make_pred(w1, b1, w2, b2, X_test, 0)
```

As the output, in the terminal I get:

```
Prediction: 7
True Digit: 7
```

and in the generated `pyplot`:

![[Screenshot 2024-03-10 at 10.31.40 PM.png| yep |300]]


*That is how you build a neural network to classify the MNIST digits from scratch, using only NumPy*