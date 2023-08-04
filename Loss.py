import numpy as np


# Calculate loss to determine how accurate our algorithm is and if any changes are needed.
class Loss:
    def calc(self, output, y):
        sample_loss = self.forward(output, y)
        data_loss = np.mean(sample_loss)
        return data_loss


# Finds the difference between the predicted and true values and returns the result.
class LossRegression(Loss):
    def forward(self, y_pred, y_true):
        correct_confidences = np.sum(abs(y_pred[i] - y_true[i]) / y_true[i] for i in range(0, len(y_true))) / len(
            y_pred)
        return correct_confidences
