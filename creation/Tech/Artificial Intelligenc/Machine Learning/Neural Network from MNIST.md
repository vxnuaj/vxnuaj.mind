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

Given that the MNIST dataset is within our directory in csv format, we use the `pd.read_csv` function to access data from the csv file.

This is assigned to train_data, and to confirm that it worked, we can print the first few rows of the dataset using the `.head` function.