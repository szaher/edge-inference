import numpy as np
import layer


class MaxPooling(layer.BaseLayer):

    def __init__(self, pool_size, strides, alpha=0.01):
        super().__init__(alpha=alpha)
        self.pool_size = pool_size
        self.strides = strides
        self.mask = []
        self.pools = []
        self.input_masked = []

    def feedforward(self, X):
        self.input = X
        depth, width, height = self.input.shape
        self.mask = np.zeros_like(self.input)
        self.input_masked = np.zeros_like(self.input)

        self.output = np.zeros((depth, int(width / self.strides), int(height / self.strides)))
        for d in range(depth):
            for i in np.arange(0, width, step=self.strides):
                for j in np.arange(0, height, step=self.strides):
                    o = self.input[d, i:i + self.strides, j:j + self.strides]
                    max_value = np.max(o)
                    omask = np.where(o == max_value, 1, 0)
                    self.mask[d, i:i + self.strides, j:j + self.strides] = omask
                    self.pools.append(max_value)

        idx = np.where(self.mask == 1)
        self.input_masked[idx] = self.input[idx]
        pools = np.array(self.pools).reshape(
            (int(self.input.shape[1] / self.strides), int(self.input.shape[2] / self.strides)))
        return pools

    def backpropagation(self, gradient):
        print(np.array(gradient).shape)
        return self.input_masked
