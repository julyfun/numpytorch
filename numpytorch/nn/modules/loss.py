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


# class CrossEntropyLoss(_Loss):
#     def forward(self, y_pred, y_true):
#         y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
#         return - np.sum(y_true * np.log(y_pred), axis=-1)

#     def backward(self, y_pred, y_true):
#         self.dout = y_pred - y_true
#         return self.dout

class CrossEntropyLoss(_Loss):
    def forward(self, y_pred, y_true):
        # 计算最大值用于数值稳定
        self.max_o = np.max(y_pred, axis=1, keepdims=True)

        # 计算对数softmax
        log_softmax = y_pred - self.max_o - \
            np.log(np.sum(np.exp(y_pred - self.max_o), axis=1, keepdims=True))

        # 计算交叉熵损失
        retval = -log_softmax[np.arange(y_true.shape[0]), y_true.reshape(-1)]

        return retval

    def backward(self, y_pred, y_true):
        # 计算softmax
        softmax = np.exp(y_pred - self.max_o) / \
            np.sum(np.exp(y_pred - self.max_o), axis=1, keepdims=True)

        # 创建梯度矩阵
        grad = softmax
        grad[np.arange(y_true.shape[0]), y_true.reshape(-1)] -= 1

        # 返回平均梯度
        return grad / y_true.shape[0]


class CrossEntropyLoss2(_Loss):
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # y_true
        # print('y_true')
        # print(y_true)
        # print(y_true.reshape(-1))
        # 这 shape 处理真逆天啊

        retval = - \
            np.log(y_pred[np.arange(y_true.shape[0]), y_true.reshape(-1)])
        # print('idx:')
        # print(y_pred[np.arange(y_true.shape[0]), y_true.reshape(-1)])
        return retval

    def backward(self, y_pred, y_true):
        self.dout = y_pred.copy()
        self.dout[np.arange(y_true.shape[0]), y_true.reshape(-1)] -= 1
        return self.dout
