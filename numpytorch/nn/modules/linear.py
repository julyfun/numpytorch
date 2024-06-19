import numpy as np
import os
from .module import Module, Layer


class Flatten(Layer):
    def __init__(self):
        pass

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dvalues):
        self.dout = dvalues.reshape(self.input_shape)
        return self.dout

    def __repr__(self):
        return f"Flatten"


class Linear(Layer):
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.weights = np.random.randn(
            n_outputs, n_inputs) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((1, n_outputs))

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights.T) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        return np.dot(dvalues, self.weights)

    def __repr__(self):
        return f"Dense({self.n_inputs}, {self.n_outputs})"
