import numpy as np
import math
import losses


class Network(object):

    def __init__(self, architecture=[], epochs=1000, batch_size=64, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.architecture = architecture
        self.cost_values = []

    def loss(self, y_true, y_pred):
        # print("yTrue >>> ", y_true.shape)
        # print("yPred >>> ", y_pred.shape)
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

    def loss_prime(self, y_true, y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

    def cost(self):
        pass

    def predict2(self, x: np.ndarray):
        a = x
        for layer in self.architecture:
            a = layer.feedforward(a)
        return np.array(a)

    def predict(self, x: np.ndarray):
        preds = []
        for a in x:
            for layer in self.architecture:
                a = layer.feedforward(a)
            preds.append(a)
        return np.array(preds)

    def fit(self, X: np.ndarray, Y: np.ndarray):

        n_samples = X.shape[0]
        _mod = n_samples * 0.05

        epochs = math.ceil(n_samples / self.batch_size)
        for epoch in range(self.epochs):
            # A = np.array([])
            _idx = 0
            print(f"Epoch {epoch}  [", end="")
            for _x, _y in zip(X, Y):
                _idx += 1
                a = _x
                # print(a.shape)
                # loop over all layers and feed forward.
                for l in self.architecture:
                    a = l.feedforward(a)
                # calculate loss
                loss = losses.mse(y_true=_y, y_pred=a)
                # A = np.append(A, a)

                # backpropagating
                dz = losses.mse_prime(y_true=_y, y_pred=a)

                for l in reversed(self.architecture):
                    dz = l.backpropagation(dz)

                if _idx % _mod == 0:
                    print("=", end="")
                    self.cost_values.append(loss)
            print("]")
            # for epoch in range(epochs):
            #     #                 print("Iteration:", itr, "  epoch ", epoch)
            #     # calculate start and stop
            #     start = epoch * self.batch_size
            #     stop = start + self.batch_size
            #     # print(f"Start {start} {stop}")
            #     # get samples of current epoch
            #     xSamples = X[start:stop]
            #     ySamples = Y[start:stop]
            #     # print("FIT X", X.shape)
            #     # print("FIT X", xSamples.shape)
            #     for _x, _y in zip(xSamples, ySamples):
            #         print("Shape of _X", _x.shape)
            #         print("Shape of _y", _y.shape)
            #         print("--------------------")
            #         a = _x
            #         # print(a.shape)
            #         # loop over all layers and feed forward.
            #         for l in self.architecture:
            #             a = l.feedforward(a)
            #         # calculate loss
            #         loss = self.loss(y_true=_y, y_pred=a)
            #         # A = np.append(A, a)
            #
            #         # backpropagating
            #         dz = loss
            #         for l in reversed(self.architecture):
            #             dz = l.backpropagation(dz)
            #         print(f"Iter: {itr} - Epoch: {epoch} - Cost: {loss} - Accuracy: ...")
            # A = A.T.reshape(Y.shape)
            # j = self.cost(y_true=Y, y_pred=A)
            # if itr % 100 == 0:
            #     print(f"Cost[{itr}]: ", j)

