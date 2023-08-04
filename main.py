# Python Neural Network written by Oneeb Aamer.
import math

from math import e as E
import nnfs
import numpy as np
import matplotlib
import sys
from nnfs.datasets import spiral_data

np.random.seed(0)

# 2a + 3b
X = [[2, 3],
     [5, 4],
     [1, 2],
     [6, 1]]
y = [2 * x[0] + 3 * x[1] for x in X]


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Loss:
    def calc(self, output, y):
        sample_loss = self.forward(output, y)
        data_loss = np.mean(sample_loss)
        return data_loss


class Loss_Regression(Loss):
    def forward(self, y_pred, y_true):
        correct_confidences = np.sum(abs(y_pred[i] - y_true[i])/y_true[i] for i in range(0,len(y_true))) / len(y_pred)
        return correct_confidences


layer1 = Layer(2, 4)
activation1 = Activation_ReLU()

layer2 = Layer(4, 1)
activation2 = Activation_ReLU()

loss_function = Loss_Regression()

lowest_loss = 999999
best_layer1_weights = layer1.weights.copy()
best_layer1_biases = layer1.biases.copy()
best_layer2_weights = layer2.weights.copy()
best_layer2_biases = layer2.biases.copy()
for iteration in range(20000):
    layer1.weights += 0.01 * np.random.randn(2, 4)
    layer1.biases += 0.01 * np.random.randn(1, 4)
    layer2.weights += 0.01 * np.random.randn(4, 1)
    layer2.biases += 0.01 * np.random.randn(1, 1)

    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    loss = loss_function.calc(activation2.output, y)

    if loss < lowest_loss:
        print("new low loss at iteration:", iteration, "accuracy: " + str(100 - round(loss * 100)) + "%, activation output:", activation2.output)
        best_layer1_weights = layer1.weights.copy()
        best_layer1_biases = layer1.biases.copy()
        best_layer2_weights = layer2.weights.copy()
        best_layer2_biases = layer2.biases.copy()
        lowest_loss = loss
