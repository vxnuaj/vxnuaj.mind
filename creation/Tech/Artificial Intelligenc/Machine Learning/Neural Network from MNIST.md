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

Our network will have an input layer of 784 Neurons, a hidden layer of 32 neurons, hence `W1`, as it represents the weighted matrix of the hidden layer taking in `784` neurons and outputting `32` value