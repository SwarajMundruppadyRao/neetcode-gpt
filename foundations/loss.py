import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)
        positive = y_true * np.log(y_pred)
        negative = (1-y_true) * np.log((1-y_pred))
        return -round((sum(positive)+sum(negative))/len(y_true), 4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        sums = 0 
        for i in range(len(y_true)):
            predictions = np.clip(y_pred[i], epsilon, 1-epsilon)
            print(predictions)
            print(y_true[i])
            losses = y_true[i] * np.log(predictions)
            print(losses)
            sums += sum(losses)
            print(sums)
        return -round(sums/len(y_true), 4)