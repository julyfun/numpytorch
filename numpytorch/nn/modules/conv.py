import numpy as np
import os
from .module import Module, Layer


class Conv2d(Layer):
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

        # Perform matrix multiplication.
        out = np.dot(w_col, X_col.T) + b_col

        # Reshape back matrix to image.
        out = np.array(np.hsplit(out, in_n)).reshape((in_n, n_C, n_H, n_W))

        self.cache = X, X_col, w_col
        return out

    def backward(self, dout):
        X, X_col, w_col = self.cache
        in_n, _, _, _ = X.shape

        self.dbiases = np.sum(dout, axis=(0, 2, 3))

        dout = dout.reshape(
            dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, in_n))
        dout = np.concatenate(dout, axis=-1)

        dX_col = np.dot(w_col.T, dout)
        dw_col = np.dot(dout, X_col)
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
        for i in range(out_h):
            for j in range(out_w):
                h_start, w_start = i * stride, j * stride
                h_end, w_end = h_start + filter_h, w_start + filter_w
                X_padded[:, :, h_start:h_end,
                         w_start:w_end] += col_reshaped[:, i, j]

        if pad == 0:
            return X_padded
        elif type(pad) is int:
            return X_padded[pad:-pad, pad:-pad, :, :]

    def __repr__(self):
        return f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


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
        out = np.einsum('ik,jk->ij', w_col, X_col) + b_col

        # Reshape back matrix to image
        out = out.reshape((in_n, n_C, n_H, n_W))

        self.cache = X, X_col, w_col
        return out

    def backward(self, dout):
        X, X_col, w_col = self.cache
        in_n, _, _, _ = X.shape

        self.dbiases = np.sum(dout, axis=(0, 2, 3))

        dout_reshaped = dout.reshape(in_n, -1)

        dX_col = np.dot(w_col.T, dout_reshaped.T)
        dw_col = np.dot(dout_reshaped, X_col)
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

        # Use array slicing and broadcasting to update X_padded
        h_start = np.arange(out_h)[:, np.newaxis] * stride
        w_start = np.arange(out_w) * stride
        X_padded[:, :, h_start[:, np.newaxis, np.newaxis] + np.arange(filter_h),
                 w_start[np.newaxis, :, np.newaxis] + np.arange(filter_w)] += col_reshaped

        if pad == 0:
            return X_padded
        elif type(pad) is int:
            return X_padded[:, :, pad:-pad, pad:-pad]

    def __repr__(self):
        return f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"

class Conv2dV3(Layer):
    
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

    def forward(self, X):
        n, c, h, w = X.shape
        kh, kw = self.kernel_size, self.kernel_size
        ph, pw = self.padding, self.padding
        sh, sw = self.stride, self.stride

        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1

        X_padded = np.pad(X, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')[1][2][3]

        out = np.zeros((n, self.out_channels, h_out, w_out))

        for i in range(h_out):
            for j in range(w_out):
                h_start, w_start = i * sh, j * sw
                h_end, w_end = h_start + kh, w_start + kw
                X_slice = X_padded[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.sum(X_slice[:, np.newaxis, :, :, :] * self.weights[np.newaxis, :, :, :, :], axis=(2, 3, 4)) + self.biases

        self.cache = X, X_padded
        return out

    def backward(self, dout):
        X, X_padded = self.cache
        n, _, h_out, w_out = dout.shape
        _, c, h, w = X.shape
        kh, kw = self.kernel_size, self.kernel_size
        ph, pw = self.padding, self.padding
        sh, sw = self.stride, self.stride

        dX_padded = np.zeros_like(X_padded)

        for i in range(h_out):
            for j in range(w_out):
                h_start, w_start = i * sh, j * sw 
                h_end, w_end = h_start + kh, w_start + kw
                dout_slice = dout[:, :, i, j][:, :, np.newaxis, np.newaxis, np.newaxis]
                dX_padded[:, :, h_start:h_end, w_start:w_end] += np.sum(dout_slice * self.weights[np.newaxis, :, :, :, :], axis=1)
                self.dweights += np.sum(X_padded[:, :, h_start:h_end, w_start:w_end][:, np.newaxis, :, :, :] * dout_slice, axis=0)

        self.dbiases = np.sum(dout, axis=(0, 2, 3))

        if self.padding > 0:
            dX = dX_padded[:, :, ph:-ph, pw:-pw]
        else:
            dX = dX_padded

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
        
        print('dout', dout.shape)
        dout = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        
        self.dbiases = np.einsum('ij->i', dout, optimize=True)
        # self.dweights = np.einsum('ij,jk->ik', dout, X_col.T.reshape((-1, X_col.T.shape[-1])), optimize=True)
        # self.dweights = np.einsum('ij,jk->ik', dout, X_col.T, optimize=True) 
        print('shape')
        print(dout.shape)
        print(X_col.shape)
        print(X_col.transpose(2, 0, 1).reshape(-1, n).shape)
        self.dweights = np.einsum('ij,jk->ik', dout, X_col.transpose(2, 0, 1).reshape(-1, c * kh * kw), optimize=True) 
        self.dweights = self.dweights.reshape(self.weights.shape)
        
        dX_col = W_row.T @ dout
        dX_col = dX_col.reshape(n, self.in_channels, self.kernel_size, self.kernel_size, dout.shape[1] // n, -1)
        print('#2')
        print(dX_col.shape)
        print(dX.shape)
        print(self.stride)
        
        dX = np.zeros_like(X)
        np.add.at(dX, (slice(None), slice(None), 
                       slice(0, None, self.stride), 
                       slice(0, None, self.stride)), dX_col)
        
        self.weights_update = np.einsum('ijklmn, ijko -> lmno', dX_col, self.delta, optimize=True)
        return dX
