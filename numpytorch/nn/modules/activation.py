import numpy as np


class Activation:
    def __init__(self):
        pass

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return self.__class__.__name__


class GeLU(Activation):
    def forward(self, x):
        self.x = x
        return x * 0.5 * (1 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))))

    def backward(self, dout):
        return dout * (0.5 * (1 + np.tanh((np.sqrt(2 / np.pi) * (self.x + 0.044715 * np.power(self.x, 3)))))) * (1 - 0.5 * np.power(np.tanh((np.sqrt(2 / np.pi) * (self.x + 0.044715 * np.power(self.x, 3)))), 2))
