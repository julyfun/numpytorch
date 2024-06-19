import numpy as np


class _Loss:
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError


class MSELoss(_Loss):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return np.mean(np.square(diff))

    def backward(self, y_pred, y_true):
        self.dout = 2 * (y_pred - y_true) / y_pred.shape[0]
        return self.dout
