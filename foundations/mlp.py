import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        a = x  # initial activation (input layer)

        for i in range(len(weights)):
            z = a @ weights[i] + biases[i]

            # Apply ReLU for all but last layer
            if i < len(weights) - 1:
                a = np.maximum(0, z)
            else:
                a = z  # output layer (no activation)

        return np.round(a, 5)
