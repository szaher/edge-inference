import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time

# allow relative import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from lib import distributed_layers
from lib import convolutional
from lib import models

# tf.enable_eager_execution(config=None, device_policy=None, execution_mode=None)
# custom_objects = {"DistributedDense": distributed_layers.DistributedDense}
custom_objects = {"DistributedDense": distributed_layers.DistributedDense, 'DistributedConv2D':
convolutional.DistributedConv2D}

host = models.AgentHost(host="http://localhost:8888", protocol="http", path="model/", model="mnist.latest", method="POST")

model = tf.keras.models.load_model("saved_models/mnist.latest.h5", custom_objects=custom_objects)

# model.layers[6].set_agent_host(host=host)

for l in model.layers:
    if hasattr(l, "set_agent_host"):
        l.set_agent_host(host)

digits = 10
input_shape = (28, 28, 1)
# set batch size and iterations
batch_size = 64
epochs = 10
# model_save_path = "../saved_models/mnist.latest.h5"

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

# import importlib
# distributed_layers = importlib.reload(distributed_layers)
#
# output = model.predict(x_test)
start = time.time()
output = model.predict(x_test[0:1])
end = time.time()

print(output)
print("Classification Time: ", end-start)
