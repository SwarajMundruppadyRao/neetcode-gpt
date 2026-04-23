import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]], b1: List[float],
        W2: List[List[float]], b2: List[float],
        y_true: List[float]
    ) -> dict:
        
        # Convert to numpy arrays
        x = np.array(x)
        w1 = np.array(W1)
        w2 = np.array(W2)
        b1 = np.array(b1)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # ---- Forward ----
        # W1: (hidden_dim, input_dim)
        z1 = w1 @ x + b1
        a1 = np.maximum(0, z1)

        # W2: (output_dim, hidden_dim)
        z2 = w2 @ a1 + b2
        y_pred = z2

        # Loss (MSE)
        n = y_true.shape[0]
        loss = np.mean((y_pred - y_true) ** 2)

        # ---- Backward ----
        dz2 = (2 / n) * (y_pred - y_true)

        # Gradients for second layer
        dW2 = np.outer(dz2, a1)   # (output_dim, hidden_dim)
        db2 = dz2

        # Backprop to hidden layer
        dA1 = w2.T @ dz2
        dz1 = dA1 * (z1 > 0)

        # Gradients for first layer
        dW1 = np.outer(dz1, x)    # (hidden_dim, input_dim)
        db1 = dz1

        # ---- Clean rounding (remove -0.0) ----
        def clean(arr):
            arr = np.round(arr, 4)
            arr[np.isclose(arr, 0)] = 0.0
            return arr.tolist()

        return {
            'loss': round(float(loss), 4),
            'dW1': clean(dW1),
            'db1': clean(db1),
            'dW2': clean(dW2),
            'db2': clean(db2)
        }