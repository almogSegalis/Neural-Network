# Neural Network

This is a simple implementation of a neural network using Python and scikit-learn library.

## Introduction

The repository contains a Python file neural_network.py, which contains the code to build a neural network using the scikit-learn library. The neural network is based on the multi-layer perceptron (MLP) model and uses various activation functions, such as relu, sigmoid, and binary, to model the neurons.

## Requirements

- Python 3.x
- scikit-learn
- numpy
- matplotlib

## Installation

1. Clone the repository to your local machine:
```
git clone https://github.com/your_username/neural_network.git
```
2. Install the required libraries:
```
pip install -r requirements.txt
```

## Usage

You can create a neural network object using the `NeuralNetwork` class in neural_network.py. The constructor of the `NeuralNetwork` class takes the following parameters:

- `layer_activation`: list of activation functions for each layer
- `input_size`: number of dimensions of the input
- `layers_size`: list of the size of each layer
- `batch_size`: number of samples that will be propagated through the network
- `learning_rate`: the learning rate of the network
- `random_state`: the seed for the random number generator

Once you have created a neural network object, you can train the network using the `fit` method and make predictions using the `predict` method.

## Example

```python
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork, Relu, Sigmoid, Binary

# Load the iris dataset
iris = load_iris()

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create a neural network object with 4 layers
nn = NeuralNetwork(layer_activation=[Relu(), Sigmoid(), Binary()], input_size=4, layers_size=[8, 6, 4], batch_size=16, learning_rate=0.001)

# Train the network using the train set
nn.fit(X_train, y_train, num_epochs=100)

# Make predictions on the test set
y_pred = nn.predict(X_test)

# Print the accuracy of the predictions
print("Accuracy:", np.mean(y_pred == y_test))
```
## License

This project is licensed under the MIT License - see the LICENSE file for details.
