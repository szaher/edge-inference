import argparse
import sys

import tensorflow as tf
import distributed_layers
import convolutional
import models


def load_model(model_path, host=None):
    custom_objects = {
        "DistributedDense": distributed_layers.DistributedDense,
        'DistributedConv2D': convolutional.DistributedConv2D
    }

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    if host:
        for layer in model.layers:
            if hasattr(layer, "set_agent_host"):
                layer.set_agent_host(host)
    return model


def convert(model, output):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('{}.tflite'.format(output), 'wb') as f:
        f.write(tflite_model)
    f.close()


def parse_args(args=[]):
    parser = argparse.ArgumentParser(description='Capstone: Distributed Edge Inference TFLite Converter')
    parser.add_argument("--distributed", default=False, action="store_true", help="Is the model distributed?")
    parser.add_argument("--url", help="Target base url to offload inference")
    parser.add_argument("--model", help="Absolute path to model h5 file.")
    parser.add_argument("-o", "--output", help="output tflite model name to be created!")
    return parser.parse_args(args)


def main():
    args = parse_args(args=sys.argv[1:])
    host = None
    if args.distributed:
        host = models.AgentHost(host=args.url, protocol="http", path="model/", model=args.model, method="POST")

    model = load_model(model_path=args.model, host=host)
    convert(model=model, output=args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
