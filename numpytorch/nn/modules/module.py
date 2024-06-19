import numpy as np
import os


class Module:
    def __init__(self):
        self.layers = []

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def add(self, layer):
        self.layers.append(layer)

    def save(self, filename):
        data = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                data[f"layer_{i}_weights"] = layer.weights
                data[f"layer_{i}_biases"] = layer.biases

        data['summary'] = self.summary()
        np.savez(filename, **data)

    def load(self, filename):
        if os.path.exists(filename):
            data = np.load(filename)
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'weights'):
                    layer.weights = data[f"layer_{i}_weights"]
                    layer.biases = data[f"layer_{i}_biases"]
            return data['summary']
        else:
            raise FileNotFoundError(f"No file found at {filename}")

    def state_dict(self):
        data = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                data[f"layer_{i}_weights"] = layer.weights
                data[f"layer_{i}_biases"] = layer.biases
        return data

    def load_state_dict(self, data):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                layer.weights = data[f"layer_{i}_weights"]
                layer.biases = data[f"layer_{i}_biases"]
        return data['summary']

    def summary(self):
        return "\n".join([str(layer) for layer in self.layers])

    @staticmethod
    def load_summary(file):
        return np.load(file)['summary']


class Layer:
    def __init__(self):
        self.weights = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __call__(self, x):
        try:
            return self.forward(x)
        except ValueError as e:
            print(
                f"Expected shape: {self.weights.shape} But received shape: {x.shape}")


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
