import numpy as np
import layer


class Dense(layer.BaseLayer):

    def __init__(self, input_shape, output_shape, alpha=0.01):
        super(Dense, self).__init__(alpha=alpha)
        self.weights = np.random.randn(output_shape, input_shape)
        self.bias = np.random.randn(output_shape, 1)

    def feedforward(self, x: np.ndarray):
        self.input = x
        # print(f"Dense Forward )>> {X.shape}")
        return np.dot(self.weights, self.input) + self.bias

    def backpropagation(self, gradient):
        # print("Back prop", gradient.shape)
        weights_gradient = np.dot(gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, gradient)
        self.weights -= self.alpha * weights_gradient
        self.bias -= self.alpha * gradient
        return input_gradient
