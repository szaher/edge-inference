import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# allow relative import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from lib import distributed_layers
from lib import convolutional
from lib import models

# tf.enable_eager_execution(config=None, device_policy=None, execution_mode=None)
# custom_objects = {"DistributedDense": distributed_layers.DistributedDense}


def parse_args(args=[]):
    parser = argparse.ArgumentParser(description='Capstone: Distributed Edge Inference')
    parser.add_argument("--host", help="Host to offload inference. Example http://example.com")
    parser.add_argument("--input", help="Input image.")
    parser.add_argument("--model", help="model name in saved_models.")
    parser.add_argument("--port", default=8888, help="Host to offload inference")
    parser.add_argument("--topk", default=1, help="Top k probability.")

    return parser.parse_args(args)


def classify(model_name, image, host=None, port=None, top=1):
    custom_objects = {
        "DistributedDense": distributed_layers.DistributedDense,
        'DistributedConv2D': convolutional.DistributedConv2D
    }
    model = tf.keras.models.load_model("saved_models/{}.h5".format(model_name), custom_objects=custom_objects)

    if host:
        host = models.AgentHost(
            host="{host}:{port}".format(host=host, port=port), protocol="http", path="model/",
            model=model_name, method="POST"
        )
        for l in model.layers:
            if hasattr(l, "set_agent_host"):
                l.set_agent_host(host)

    img = Image.open(image)

    # convert image to numpy array
    data = np.asarray(img, dtype='float32')

    # normalize...
    input_shape = list(model.input_shape)
    input_shape[0] = 1

    data = data.reshape(tuple(input_shape)) / 255.

    start = time.time()

    output = model.predict(data)

    end = time.time()

    idx = np.argpartition(output.flatten(), kth=-top)[-top:]

    print(f"Execution time: {end - start}")

    return idx


def main():
    args = parse_args(sys.argv[1:])
    classify(model_name=args.model, image=args.input, host=args.host, port=int(args.port), top=int(args.topk))
    return 0


if __name__ == '__main__':
    sys.exit(main())
