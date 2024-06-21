import numpy as np
import os
from .module import Module, Layer
import scipy
import scipy.signal

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding

        self.weights = np.random.randn(self.out_channels, self.in_channels,
                                       self.kernel_size, self.kernel_size) * np.sqrt(1. / (self.kernel_size))
        self.biases = np.random.randn(self.out_channels) * np.sqrt(1. / self.out_channels)

        self.dweights = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.dbiases = np.zeros((self.out_channels))

        self.cache = None

    def im2col(self, X):
        n, c, h, w = X.shape
        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        X_padded = np.pad(X, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')
        
        i0 = np.repeat(np.arange(self.kernel_size), self.kernel_size)
        i0 = np.tile(i0, c)
        i1 = self.stride * np.repeat(np.arange(out_h), out_w)
        j0 = np.tile(np.arange(self.kernel_size), self.kernel_size * c)
        j1 = self.stride * np.tile(np.arange(out_w), out_h)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        
        k = np.repeat(np.arange(c), self.kernel_size * self.kernel_size).reshape(-1, 1)
        
        cols = X_padded[:, k, i, j]
        cols = cols.transpose(1, 2, 0).reshape(self.kernel_size * self.kernel_size * c, -1)
        return cols

    def col2im(self, cols, X_shape):
        n, c, h, w = X_shape
        h_padded, w_padded = h + 2 * self.padding, w + 2 * self.padding
        X_padded = np.zeros((n, c, h_padded, w_padded), dtype=cols.dtype)
        
        k, i, j = self.get_im2col_indices(X_shape)
        cols_reshaped = cols.reshape(c * self.kernel_size * self.kernel_size, -1, n)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(X_padded, (slice(None), k, i, j), cols_reshaped)
        
        if self.padding == 0:
            return X_padded
        return X_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

    def get_im2col_indices(self, X_shape):
        n, c, h, w = X_shape
        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        i0 = np.repeat(np.arange(self.kernel_size), self.kernel_size)
        i0 = np.tile(i0, c)
        i1 = self.stride * np.repeat(np.arange(out_h), out_w)
        j0 = np.tile(np.arange(self.kernel_size), self.kernel_size * c)
        j1 = self.stride * np.tile(np.arange(out_w), out_h)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(c), self.kernel_size * self.kernel_size).reshape(-1, 1)

        return (k, i, j)

    def forward(self, X):
        n, c, h, w = X.shape
        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        X_col = self.im2col(X)
        W_col = self.weights.reshape(self.out_channels, -1)
        
        output = np.dot(W_col, X_col) + self.biases.reshape(-1, 1)
        output = output.reshape(self.out_channels, out_h, out_w, n)
        output = output.transpose(3, 0, 1, 2)
        
        self.cache = X, X_col
        return output

    def backward(self, dout):
        X, X_col = self.cache
        n, c, h, w = X.shape
        
        db = np.sum(dout, axis=(0, 2, 3))
        
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        dW = np.dot(dout_reshaped, X_col.T)
        dW = dW.reshape(self.weights.shape)
        
        W_reshape = self.weights.reshape(self.out_channels, -1)
        dX_col = np.dot(W_reshape.T, dout_reshaped)
        dX = self.col2im(dX_col, X.shape)
        
        self.dweights = dW
        self.dbiases = db
        
        return dX
class Conv2dV3(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding

        self.weights = np.random.randn(self.out_channels, self.in_channels,
                                       self.kernel_size, self.kernel_size) * np.sqrt(1. / (self.kernel_size))
        self.biases = np.random.randn(
            self.out_channels) * np.sqrt(1. / self.out_channels)

        self.dweights = np.zeros(
            (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.dbiases = np.zeros((self.out_channels))

        self.cache = None

    def forward(self, X):
        n, c, h, w = X.shape
        n_C, _, kh, kw = self.weights.shape
        
        h_out = (h + 2 * self.padding - kh) // self.stride + 1
        w_out = (w + 2 * self.padding - kw) // self.stride + 1
        
        X_col = np.lib.stride_tricks.as_strided(X, shape=(n, c, kh, kw, h_out, w_out), 
                    strides=(X.strides[0], X.strides[1], X.strides[2] * self.stride, 
                             X.strides[3] * self.stride, X.strides[2], X.strides[3]))
        X_col = X_col.reshape(n, c * kh * kw, h_out * w_out)
        
        W_row = self.weights.reshape(n_C, -1)
        
        out = W_row @ X_col + self.biases.reshape(-1, 1)
        out = out.reshape(n_C, h_out, w_out, n)
        out = out.transpose(3, 0, 1, 2)
        
        self.cache = X, X_col, W_row
        return out

    def backward(self, dout):
        X, X_col, W_row = self.cache
        n, _, h, w = X.shape
        
        dout = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        
        print(dout.shape, X_col.shape)
        print(X_col.transpose(1, 2, 0).reshape(-1, h * w * n).shape)
        self.dbiases = np.sum(dout, axis=1)
        self.dweights = (dout @ X_col.transpose(1, 2, 0).reshape(-1, h * w * n).T).reshape(self.weights.shape)
        
        print('W_row', W_row.shape, 'dout', dout.shape)
        dX_col = W_row.T @ dout
        dX_col = dX_col.reshape(n, self.in_channels, self.kernel_size, self.kernel_size, dout.shape[1] // (n), -1)
        
        pad = self.padding
        dX = np.zeros((n, self.in_channels, X.shape[2] + 2*pad, X.shape[3] + 2*pad))
        # shape
        # (32, 1, 30, 30)
        print(dX.shape)
        # (32, 1, 3, 3, 784, 1)
        print(dX_col.shape)
        dX[:, :, pad:X.shape[2]+pad, pad:X.shape[3]+pad] = np.einsum('nijklm->niklm', dX_col)
        
        if pad > 0:
            dX = dX[:, :, pad:-pad, pad:-pad]
        
        return dX

    def __repr__(self):
        return f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"

class Conv2dV4(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.out_channels = out_channels
        self.kernel_size = kernel_size 
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        
        self.weights = np.random.randn(self.out_channels, self.in_channels, 
                                       self.kernel_size, self.kernel_size) * np.sqrt(1. / (self.kernel_size))
        self.biases = np.random.randn(self.out_channels) * np.sqrt(1. / self.out_channels)
        self.dweights = np.zeros_like(self.weights)  
        self.dbiases = np.zeros_like(self.biases)
        self.cache = None

    def forward(self, X):
        n, c, h, w = X.shape
        n_C, _, kh, kw = self.weights.shape
        
        h_out = (h + 2 * self.padding - kh) // self.stride + 1
        w_out = (w + 2 * self.padding - kw) // self.stride + 1
        
        X_col = np.lib.stride_tricks.as_strided(X, shape=(n, c, kh, kw, h_out, w_out), 
                    strides=(X.strides[0], X.strides[1], X.strides[2] * self.stride, 
                             X.strides[3] * self.stride, X.strides[2], X.strides[3]))
        X_col = X_col.reshape(n, c * kh * kw, h_out * w_out)
        
        W_row = self.weights.reshape(n_C, -1)
        
        out = W_row @ X_col + self.biases.reshape(-1, 1)
        out = out.reshape(n_C, h_out, w_out, n)
        out = out.transpose(3, 0, 1, 2)
        
        self.cache = X, X_col, W_row
        return out

    def backward(self, dout):
        X, X_col, W_row = self.cache
        n, c, h, w = X.shape
        n_C, _, kh, kw = self.weights.shape
        
        # print('dout', dout.shape)
        dout = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        
        self.dbiases = np.einsum('ij->i', dout, optimize=True)
        # self.dweights = np.einsum('ij,jk->ik', dout, X_col.T.reshape((-1, X_col.T.shape[-1])), optimize=True)
        # self.dweights = np.einsum('ij,jk->ik', dout, X_col.T, optimize=True) 
        # print('shape')
        # print(dout.shape)
        # print(X_col.shape)
        # print(X_col.transpose(2, 0, 1).reshape(-1, n).shape)
        self.dweights = np.einsum('ij,jk->ik', dout, X_col.transpose(2, 0, 1).reshape(-1, c * kh * kw), optimize=True) 
        self.dweights = self.dweights.reshape(self.weights.shape)
        W_row = self.weights.reshape(n_C, -1)
        dX_col = W_row.T @ dout
        dX_col = dX_col.reshape(n, c, kh, kw, h, w)
        
        dX = np.zeros_like(X)
                
        return dX

class Conv2dV2(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding

        self.weights = np.random.randn(self.out_channels, self.in_channels,
                                       self.kernel_size, self.kernel_size) * np.sqrt(1. / (self.kernel_size))
        self.biases = np.random.randn(
            self.out_channels) * np.sqrt(1. / self.out_channels)

        self.dweights = np.zeros(
            (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.dbiases = np.zeros((self.out_channels))

        self.cache = None

    def forward(self, X):
        in_n, in_c, in_h, in_w = X.shape

        n_C = self.out_channels
        n_H = int((in_h + 2 * self.padding - self.kernel_size) / self.stride) + 1
        n_W = int((in_w + 2 * self.padding - self.kernel_size) / self.stride) + 1

        X_col = self._im2col(X, self.kernel_size,
                             self.kernel_size, self.stride, self.padding)
        w_col = self.weights.reshape((self.out_channels, -1))
        b_col = self.biases.reshape(-1, 1)

        # Perform matrix multiplication using np.einsum
        # shape
        # print (w_col.shape, X_col.shape)
        out = np.einsum('ij,jk->ik', w_col, X_col.T) + b_col

        # Reshape back matrix to image using np.reshape
        out = out.reshape((in_n, n_C, n_H, n_W))

        self.cache = X, X_col, w_col
        return out

    def backward(self, dout):
        X, X_col, w_col = self.cache
        in_n, _, _, _ = X.shape

        self.dbiases = np.sum(dout, axis=(0, 2, 3))

        # dout_reshaped = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = dout.reshape(
            dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, in_n))
        dout = np.concatenate(dout, axis=-1)

        dX_col = np.einsum('ij,jk->ik', w_col.T, dout)
        dw_col = np.einsum('ij,jk->ik', dout, X_col)
        # print(dX_col.shape, dw_col.shape)
        dX = self._col2im(dX_col, X.shape, self.kernel_size,
                          self.kernel_size, self.stride, self.padding)

        self.dweights = dw_col.reshape(
            (dw_col.shape[0], self.in_channels, self.kernel_size, self.kernel_size))

        return dX

    def _im2col(self, input_data, filter_h, filter_w, stride, pad):
        img = np.pad(input_data, [(0, 0), (0, 0),
                     (pad, pad), (pad, pad)], 'constant')
        N, C, H, W = img.shape
        NN, CC, HH, WW = img.strides
        out_h = (H - filter_h)//stride + 1
        out_w = (W - filter_w)//stride + 1
        col = np.lib.stride_tricks.as_strided(
            img, (N, out_h, out_w, C, filter_h, filter_w), (NN, stride * HH, stride * WW, CC, HH, WW)).astype(float)
        return col.reshape(np.multiply.reduceat(col.shape, (0, 3)))

    def _col2im(self, col, input_shape, filter_h, filter_w, stride, pad):
        N, C, H, W = input_shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        col_reshaped = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
        X_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=col.dtype)
        
        h_start = np.arange(out_h)[:, np.newaxis] * stride
        w_start = np.arange(out_w) * stride
        h_end = h_start + filter_h
        w_end = w_start + filter_w
        
        np.add.at(X_padded, (slice(None), slice(None), h_start, w_start), col_reshaped)

        raise KeyError
        if pad == 0:
            return X_padded
        elif type(pad) is int:
            return X_padded[:, :, pad:-pad, pad:-pad]
    