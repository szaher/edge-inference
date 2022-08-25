import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# allow relative import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from lib.distributed_layers import DistributedDense
from lib import models
from lib import convolutional


def plot_sample(train_images, class_names, train_labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()


def main():
    # tf.enable_eager_execution(config=None, device_policy=None, execution_mode=None)
    classes = 10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    input_shape = (32, 32, 3)
    # set batch size and iterations
    batch_size = 256
    epochs = 10
    model_save_path = "saved_models/cifar10.latest.h5"

    # Load the data and split it between train and test sets
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            convolutional.DistributedConv2D(32, kernel_size=(3, 3), activation="relu", layer_index=0),
            layers.MaxPooling2D(pool_size=(2, 2)),
            convolutional.DistributedConv2D(64, kernel_size=(3, 3), activation="relu", layer_index=2),
            layers.MaxPooling2D(pool_size=(2, 2)),
            convolutional.DistributedConv2D(64, kernel_size=(3, 3), activation="relu", layer_index=4),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            DistributedDense(64, activation="relu", layer_index=7),
            DistributedDense(classes, layer_index=8),
        ]
    )

    # Print model summary
    model.summary()

    # compile the model and use categorical cross entropy for loss as we have 10 categories
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # train the model
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))

    # evaluate the model
    print("Time to evaluate!......................")
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print("Loss Score:", test_loss)
    print("Accuracy Score:", test_acc)

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    print(">>>>>>>>>>>>>>>>>>>>>>>> Predict <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    # host = models.AgentHost(host="http://localhost:8888", protocol="http", path="model/", model="mnist.latest", method="POST")
    # model.layers[6].set_agent_host(host=host)
    #
    # output = model.predict(x_test)
    # print(output)


if __name__ == '__main__':
    sys.exit(main())
