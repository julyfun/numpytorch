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


class CrossEntropyLoss(_Loss):
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - np.sum(y_true * np.log(y_pred), axis=-1)

    def backward(self, y_pred, y_true):
        self.dout = y_pred - y_true
        return self.dout


class CrossEntropyLoss2(_Loss):
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.log(y_pred[np.arange(y_true.shape[0]), y_true])

    def backward(self, y_pred, y_true):
        self.dout = y_pred.copy()
        self.dout[np.arange(y_true.shape[0]), y_true] -= 1
        return self.dout
