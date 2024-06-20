from .optimizer import Optimizer
import numpy as np


class SGD(Optimizer):
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.lr = lr

    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights -= self.lr * layer.dweights.T
                layer.biases -= self.lr * layer.dbiases
