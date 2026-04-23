import numpy as np
from typing import List


class Solution:
    def Linear(self, x, W1, b1):
        return np.dot(W1, x) + b1

    def ReLU(self, z):
        return np.maximum(0, z)

    def forward_and_backward(self,
                            x: List[float],
                            W1: List[List[float]], b1: List[float],
                            W2: List[List[float]], b2: List[float],
                            y_true: List[float]) -> dict:
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        z1 = self.Linear(x, W1, b1)
        a1 = self.ReLU(z1)
        z2 = self.Linear(a1, W2, b2)

        L = np.mean((z2 - y_true) ** 2)

        dl_z2 = 2 * (z2 - y_true) / len(y_true)
        dl_w2 = dl_z2[:, None] * a1[None, :]   # make 2D
        dl_b2 = dl_z2

        dl_a1 = W2.T @ dl_z2                   # matrix multiply
        dl_z1 = dl_a1 * (z1 > 0)

        dl_w1 = dl_z1[:, None] * x[None, :]    # make 2D
        dl_b1 = dl_z1                          # keep 1D

        # remove negative zeros after rounding
        dl_w1 = np.where(np.round(dl_w1, 4) == 0, 0.0, np.round(dl_w1, 4))
        dl_b1 = np.where(np.round(dl_b1, 4) == 0, 0.0, np.round(dl_b1, 4))
        dl_w2 = np.where(np.round(dl_w2, 4) == 0, 0.0, np.round(dl_w2, 4))
        dl_b2 = np.where(np.round(dl_b2, 4) == 0, 0.0, np.round(dl_b2, 4))

        return {
            "loss": round(float(L), 4),
            "dW1": dl_w1.tolist(),
            "db1": dl_b1.tolist(),
            "dW2": dl_w2.tolist(),
            "db2": dl_b2.tolist()
        }