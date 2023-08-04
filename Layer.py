import numpy as np


# Object to store each layer of the neural network. Biases and weights are randomly generated upon initialization.
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward propagation to send data to the next layer using matrix dot products.
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
