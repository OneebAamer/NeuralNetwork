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
     [6, 1]]

# Target Values: 2a + 3b
y = [2 * x[0] + 3 * x[1] for x in X]


def run():
    layer1 = Layer(2, 4)
    activation1 = ActivationReLU()

    layer2 = Layer(4, 1)
    activation2 = ActivationReLU()

    loss_function = LossRegression()

    lowest_loss = 999999
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
            print("new low loss at iteration:", iteration,
                  "accuracy: " + str(100 - round(loss * 100)) + "%, activation output: \n", activation2.output)
            lowest_loss = loss


run()
