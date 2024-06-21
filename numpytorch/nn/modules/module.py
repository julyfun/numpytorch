import numpy as np
import os
import time


class Module:
    def __init__(self):
        self.layers = []
        self.f_timer = {}
        self.b_timer = {}

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            start_time = time.perf_counter()
            x = layer.forward(x)
            end_time = time.perf_counter()
            self.f_timer[i] = self.f_timer.get(i, 0) + (end_time - start_time)
        return x

    def backward(self, grad):
        for i, layer in zip(reversed(range(len(self.layers))), reversed(self.layers)):
            start_time = time.perf_counter()
            grad = layer.backward(grad)
            end_time = time.perf_counter()
            self.b_timer[i] = self.b_timer.get(i, 0) + (end_time - start_time)

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


class Debug(Layer):
    def __init__(self):
        pass

    def forward(self, x):
        print(f"Forward: {x.shape}")
        return x

    def backward(self, dvalues):
        print(f"Backward: {dvalues.shape}")
        return dvalues


class AveragePooling2D(Layer):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.cache = None

    def forward(self, X):
        N, C, self.H, self.W = X.shape
        kh, kw = self.pool_size, self.pool_size
        new_h = int(self.H / kh)
        new_w = int(self.W / kw)
        X_reshaped = X.reshape(N, C, kh, kw, new_h, new_w)
        out = X_reshaped.mean(axis=(2, 3))
        self.cache = (X_reshaped, out)
        return out

    def backward(self, dout):
        X_reshaped, out = self.cache
        N, C, kh, kw, new_h, new_w = X_reshaped.shape
        dX_reshaped = np.zeros_like(X_reshaped)
        dX_reshaped.reshape(N, C, kh*kw, new_h, new_w)[range(N), :, :, :, :] = \
            dout[:, :, np.newaxis, :, :] / (kh*kw)
        dX = dX_reshaped.reshape(N, C, self.H, self.W)
        return dX

    def __repr__(self):
        return f"MaxPool2D({self.pool_size})"
