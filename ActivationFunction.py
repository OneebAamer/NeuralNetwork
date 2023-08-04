import numpy as np


# Rectified Linear Unit algorithm for our activation function.
class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
