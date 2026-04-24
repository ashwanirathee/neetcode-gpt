import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        n_samples, n_features = X.shape
        w = np.zeros(n_features)

        print(w.shape, X.shape)
        b = 0
        for epoch in range(epochs):
            print(epoch)
            y_hat = X @ w + b

            L = np.mean(y_hat - y)**2
            n = n_samples
            dl_dw = 2*X.T @ (y_hat-y)/n
            dl_db = 2*np.sum(y_hat-y)/n
            w = w - lr*dl_dw
            b = b - lr*dl_db 
        
        return np.round(w, 5), np.round(b, 5)
