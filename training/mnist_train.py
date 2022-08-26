import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# allow relative import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from lib.distributed_layers import DistributedDense
from lib import models
from lib import convolutional


def main():
    # tf.enable_eager_execution(config=None, device_policy=None, execution_mode=None)
    digits = 10
    input_shape = (28, 28, 1)
    # set batch size and iterations
    batch_size = 256
    epochs = 10
    model_save_path = "saved_models/mnist.latest.h5"

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255.
    x_test = x_test.astype("float32") / 255.
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, digits)
    y_test = keras.utils.to_categorical(y_test, digits)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            convolutional.DistributedConv2D(32, kernel_size=(3, 3), activation="relu", layer_index=0),
            layers.MaxPooling2D(pool_size=(2, 2)),
            convolutional.DistributedConv2D(64, kernel_size=(3, 3), activation="relu", layer_index=2),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            DistributedDense(digits, activation="softmax", layer_index=6),
        ]
    )

    # Print model summary
    model.summary()

    # compile the model and use categorical cross entropy for loss as we have 10 categories
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    # evaluate the model
    print("Time to evaluate!......................")
    score = model.evaluate(x_test, y_test, verbose=0)

    print("Loss Score:", score[0])
    print("Accuracy Score:", score[1])

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    print(">>>>>>>>>>>>>>>>>>>>>>>> Predict <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    #
    # host = models.AgentHost(host="http://localhost:8888", protocol="http", path="model/", model="mnist.latest", method="POST")
    # model.layers[6].set_agent_host(host=host)
    #
    # output = model.predict(x_test)
    # print(output)


if __name__ == '__main__':
    sys.exit(main())
