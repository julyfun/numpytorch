import numpy as np
import os
from .module import Module, Layer


class MaxPool2d(Layer):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.cache = None

    def forward(self, X):
        self.cache = X
        n, c, h, w = X.shape
        ph, pw = self.pool_size
        oh, ow = h//ph, w//pw
        out = np.zeros((n, c, oh, ow))
        for i in range(oh):
            for j in range(ow):
                out[:, :, i, j] = np.amax(
                    X[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw], axis=(2, 3))
        return out

    def backward(self, dout):
        n, c, oh, ow = dout.shape
        ph, pw = self.pool_size
        _, _, h, w = self.cache.shape
        dx = np.zeros_like(self.cache)
        for i in range(oh):
            for j in range(ow):
                window = self.cache[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                m = np.amax(window, axis=(2, 3), keepdims=True)
                mask = (window == m)
                dx[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw] += mask * \
                    (dout[:, :, i, j])[:, :, None, None]
        return dx

    def __repr__(self):
        return f"MaxPool2D({self.pool_size})"


class AvgPool2d(Layer):
    def __init__(self, kernel_size):
        self.pool_size = kernel_size
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
