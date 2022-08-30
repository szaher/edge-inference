import os
import sys

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# allow relative import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from lib.distributed_layers import DistributedDense
from lib import distributed_layers
from lib import convolutional
from lib import models

# tf.enable_eager_execution(config=None, device_policy=None, execution_mode=None)
# custom_objects = {"DistributedDense": distributed_layers.DistributedDense}
class_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

custom_objects = {"DistributedDense": distributed_layers.DistributedDense, 'DistributedConv2D':
convolutional.DistributedConv2D}

host = models.AgentHost(host="http://localhost:8888", protocol="http", path="model/", model="cifar10.latest", method="POST")

model = tf.keras.models.load_model("saved_models/cifar10.latest.h5", custom_objects=custom_objects)

# model.layers[6].set_agent_host(host=host)
#
for l in model.layers:
    if hasattr(l, "set_agent_host"):
        l.set_agent_host(host)

# model_save_path = "../saved_models/mnist.latest.h5"

# Load the data and split it between train and test sets
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Scale images to the [0, 1] range
x_test = test_images.astype("float32") / 255.

start = time.time()
output = model.predict(x_test[0:1])
end = time.time()

idxes = np.argpartition(output[0], -3)[-3:]

print("* Probs: ", output.tolist())

for idx in idxes:
    print("> class: ", class_names[idx])

idx = np.argmax(output)
print("> Top class: ", class_names[idx])

print("Classification Time: ", end-start)
