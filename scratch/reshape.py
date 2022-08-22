import numpy as np
import layer


class Reshape(layer.BaseLayer):
    def __init__(self, input_shape, output_shape, alpha=0.01):
        # print("Reshape")
        super().__init__(alpha=alpha)
        self.input_shape = input_shape
        self.output_shape = output_shape

    def feedforward(self, x):
        self.input = x
        return np.reshape(self.input, self.output_shape)

    def backpropagation(self, gradient):
        return np.reshape(gradient, self.input_shape)
