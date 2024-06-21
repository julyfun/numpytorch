from .optimizer import Optimizer
import numpy as np


class SGD(Optimizer):
    def __init__(self, layers, lr, batch_size):
        self.layers = layers
        self.lr = lr
        self.batch_size = batch_size

    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights -= self.lr * layer.dweights.T / self.batch_size
                layer.biases -= self.lr * layer.dbiases / self.batch_size
