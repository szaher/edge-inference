import numpy as np
from scipy import signal
import math
import sys
from tensorflow import keras
from keras.utils import np_utils

# local
from dense import Dense
from reshape import Reshape
from maxpooling import MaxPooling
import model
import activations


def load_data(x, y, limit):
    # zero_index = np.where(y == 0)[0][:limit]
    # one_index = np.where(y == 1)[0][:limit]
    # all_indices = np.hstack((zero_index, one_index))
    # all_indices = np.random.permutation(all_indices)
    # x, y = x[all_indices], y[all_indices]
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255.
    # x = x.reshape(len(x), 1, 28, 28)
    # x = x.astype("float32") / 255.
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x[:limit], y[:limit]


def preprocess_data(x, y, limit):
    # zero_index = np.where(y == 0)[0][:limit]
    # one_index = np.where(y == 1)[0][:limit]
    # all_indices = np.hstack((zero_index, one_index))
    # all_indices = np.random.permutation(all_indices)
    # x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y


def preprocess_data2(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, y_train = load_data(x_train, y_train, 10000)
    x_test, y_test = load_data(x_test, y_test, 20)
    print(f"Dataset: mnist loaded.")
    print(f"Training Samples: {x_train.shape[0]}, Test Samples {x_test.shape[0]}.")
    print("X Shape ", x_train.shape)
    # print("Yzzzzz: ", y_test.shape)
    # print("Yzzzzz: ", y_test[0])
    #
    # print("Yzzzzz: ", y_test[0].shape)
    # print(y_test)
    img_shape = 1 * 28 * 28

    architecture = [
        # Reshape(input_shape=(28, 28), output_shape=img_shape),
        Dense(input_shape=img_shape, alpha=0.1, output_shape=200),
        activations.Tanh(),
        Dense(input_shape=200, alpha=0.01, output_shape=100),
        activations.Tanh(),
        Dense(input_shape=100, alpha=0.1, output_shape=20),
        activations.Tanh(),
        Dense(input_shape=20, alpha=0.1, output_shape=2),
        activations.Tanh()
    ]

    architecture = [
        Dense(28 * 28, 40),
        activations.Tanh(),
        Dense(40, 10),
        activations.Tanh()
    ]

    net = model.Network(architecture=architecture, epochs=50, batch_size=x_train.shape[0])
    net.fit(x_train, y_train)

    print("Predictions!!")
    y_pred = net.predict(x_test)
    print(y_pred)
    print(y_test.flatten())
    print(y_test.shape)
    print(y_pred.shape)
    sim = (y_test == y_pred)
    print(sim)
    print("Accuracy: >> ", len(np.where(sim == True))/ len(sim))

    return 0


if __name__ == "__main__":
    sys.exit(main())
