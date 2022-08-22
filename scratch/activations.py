import numpy as np
import layer


class Tanh(layer.BaseLayer):

    def __init__(self, alpha=0.01):
        super().__init__(alpha=alpha)

    def feedforward(self, X):
        print("===================================================")
        print(X)
        print("===================================================")
        return np.tanh(X)

    def backpropagation(self, gradient):
        print("===================================================")
        print(gradient)
        print("===================================================")
        return 1 - np.tanh(gradient) ** 2


class Sigmoid(layer.BaseLayer):

    def __init__(self, alpha=0.01):
        super(Sigmoid, self).__init__(alpha=alpha)

    def feedforward(self, x):
        self.input = x
        return 1 / (1 + np.exp(-self.input))

    def backpropagation(self, gradient):
        a = self.feedforward(self.input)
        return a * (1 - a)


class Relu(layer.BaseLayer):

    def __init__(self, alpha=0.01):
        super(Relu, self).__init__(alpha=alpha)

    def feedforward(self, x: np.ndarray):
        self.input = x
        idx = np.where(self.input < 0)
        updated = self.input.copy()
        updated[idx] = 0
        return updated

    def backpropagation(self, gradient):
        idx0 = np.where(self.input < 0)
        idx1 = np.where(self.input > 1)
        x = self.input.copy()
        x[idx0] = 0
        x[idx1] = 1
        return x
