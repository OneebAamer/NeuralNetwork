# Python Neural Network written by Oneeb Aamer.
import numpy as np
from Layer import Layer
from ActivationFunction import ActivationReLU
from Loss import LossRegression

# Change this to get new randomly generated numbers
np.random.seed(0)

# Input data
X = [[5, 4],
     [2, 3],
     [1, 2],
     [6, 1],
     [3, 3]]

# Target Values: 2a + 3b
y = [2 * x[0] + 3 * x[1] for x in X]

layer1 = Layer(2, 4)
activation1 = ActivationReLU()

layer2 = Layer(4, 1)
activation2 = ActivationReLU()

def train():
    loss_function = LossRegression()

    lowest_loss = 999999
    best_layer1_weights = layer1.weights.copy()
    best_layer1_biases = layer1.biases.copy()
    best_layer2_weights = layer2.weights.copy()
    best_layer2_biases = layer2.biases.copy()
    for iteration in range(10000):
        layer1.weights += 0.02 * np.random.randn(2, 4)
        layer1.biases += 0.02 * np.random.randn(1, 4)
        layer2.weights += 0.02 * np.random.randn(4, 1)
        layer2.biases += 0.02 * np.random.randn(1, 1)

        layer1.forward(X)
        activation1.forward(layer1.output)

        layer2.forward(activation1.output)
        activation2.forward(layer2.output)

        loss = loss_function.calc(activation2.output, y)

        if loss < lowest_loss:
            print("new low loss at iteration:", iteration,
                  "accuracy: " + str(100 - round(loss * 100)) + "%, activation output: \n", activation2.output)
            lowest_loss = loss
            best_layer1_weights = layer1.weights.copy()
            best_layer1_biases = layer1.biases.copy()
            best_layer2_weights = layer2.weights.copy()
            best_layer2_biases = layer2.biases.copy()
    layer1.weights = best_layer1_weights
    layer1.biases = best_layer1_biases
    layer2.weights = best_layer2_weights
    layer2.biases = best_layer2_biases




def run():
    train()
    a = int(input("Enter a value for a: "))
    b = int(input("Enter a value for b: "))

    layer1.forward([a, b])
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    print(activation2.output)




run()
