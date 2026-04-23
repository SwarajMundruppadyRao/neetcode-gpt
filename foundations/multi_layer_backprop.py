import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        W1 = np.array(W1) # 2x2
        b1 = np.array(b1) # 1x2
        W2 = np.array(W2) # 1x2
        b2 = np.array(b2) # 1x1
        y_true = np.array(y_true)
        
        z1 =  x @ W1.T + b1 # 1x2 * 2x2 = 1x2
        a1 = np.maximum(z1, 0) # 1x2
        z2 = a1 @ W2.T + b2 # 1x2 * 2x1 = 1x1
        loss = np.mean((z2-y_true)**2) 
        results = {}
        results['loss'] = np.round(loss, 4)

        n = len(y_true) if y_true.ndim > 0 else 1


        dz2 = 2 * (z2-y_true)/n
        dW2 = dz2.reshape(-1,1) @ a1.reshape(1,-1) # output you want is 1x2 (1x1 * 1x2)
        dB2 = dz2 
        da1 = dz2.reshape(-1,1) @ W2 # output should be 1x2 (1x1 * 1x2)
        da1 = da1.flatten()
        dz1 = da1 * (a1!=0).astype(float) # relu derivative # 1x2
        dW1 = dz1.reshape(-1, 1) @ x.reshape(1, -1) # u want 2x2 (2x1 * 1x2)
        dB1 = dz1

        results['dW2'] = np.round(dW2, 4).tolist()
        results['db2'] = np.round(dB2, 4).tolist()
        results['dW1'] = np.round(dW1, 4).tolist()
        results['db1'] = np.round(dB1, 4).tolist()
        return results


