import argparse
import os.path
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn_ops

from tensorflow import keras
from fastapi import FastAPI

import uvicorn

# from . import distributed_layers

# allow relative import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from lib import distributed_layers
from lib import convolutional
from lib import models
# static variables

MODELS_PATH = os.path.join(os.path.abspath("."), "saved_models/")

app = FastAPI(title="Helper Agent", description="This api to help others execute their DL models.")


@app.post("/model/{model_name}/matmul")
async def mat_mul(model_name, model_helper: models.ModelHelper):
    model_path = os.path.join(MODELS_PATH, "{}.h5".format(model_name))
    model = load_model(model_path=model_path)

    layer = model.layers[model_helper.layer_index]

    weights = get_weights(
        layer=layer, start=model_helper.start_index, end=model_helper.end_index
    )

    output = gen_math_ops.MatMul(a=model_helper.data, b=weights)
    # print(output.numpy())
    # print("Output Shape ", output.shape)
    return output.numpy().tolist()


@app.post("/model/{model_name}/convolv")
async def convolv(model_name, conv_helper: models.Conv2DHelper):
    model_path = os.path.join(MODELS_PATH, "{}.h5".format(model_name))
    model = load_model(model_path=model_path)
    layer = model.layers[conv_helper.layer_index]

    kernel = layer.kernel[:, :, conv_helper.kernel_index[0]:conv_helper.kernel_index[1], :]
    # data = np.array(conv_helper.data)

    outputs = nn_ops.convolution_v2(
        input=conv_helper.data,
        padding=conv_helper.padding,
        strides=conv_helper.strides,
        dilations=conv_helper.dilations,
        data_format=conv_helper.data_format,
        name="Conv2D",
        filters=kernel
    )
    return outputs.numpy().tolist()


def get_convolv_kernel():
    return np.array([])


@app.get("/status")
async def agent_status():
    return "ok"


def get_weights(layer, start: int, end: int):
    for weight in layer.weights:
        if "kernel" in weight.name:
            if end == -1:
                return weight.numpy()[start:]
            return weight.numpy()[start:end]
    return


def load_model(model_path):
    custom_objects = {
        "DistributedDense": distributed_layers.DistributedDense,
        "DistributedConv2D": convolutional.DistributedConv2D
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model


def parse_args(args=[]):
    parser = argparse.ArgumentParser(description='Capstone: Distributed Edge Inference')
    parser.add_argument("--host", default=os.environ.get("AGENT_HOST", "0.0.0.0"), help="Host to offload inference")
    parser.add_argument("--port", default=os.environ.get("AGENT_PORT", 8888), help="username to access host to offload inference")
    parser.add_argument("--model", help="Path to model h5 file.")
    parser.add_argument("-v", "--verbose", default=os.environ.get("AGENT_VERBOSE", False), action="store_true", help="Verbose output")
    return parser.parse_args(args)


def main():
    args = parse_args(args=sys.argv[1:])
    log_level = "info"
    if args.verbose:
        log_level = "debug"

    config = uvicorn.Config("main:app", host=args.host, port=int(args.port), log_level=log_level)
    server = uvicorn.Server(config)
    server.run()


if __name__ == '__main__':
    sys.exit(main())

