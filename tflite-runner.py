import os.path

from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np
import time
import argparse
import sys
import numpy


def parse_args(args=[]):
    parser = argparse.ArgumentParser(description='Capstone: Distributed Edge Inference TFLite Converter')
    parser.add_argument("--model", help="Absolute path to model .tflite file.")
    parser.add_argument("-i", "--input", help="input file to classify")
    return parser.parse_args(args)


def classify(interpreter, image, top=1):
    tensorIdx = interpreter.get_input_details()[0]['index']
    inTensor = interpreter.tensor(tensorIdx)()[0]
    inTensor[:, :] = image

    # invoke
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

    ordered = np.argpartition(-output, 1)
    return [(i, output[i]) for i in ordered[:top]][0]


def main():
    args = parse_args(args=sys.argv[1:])

    if not os.path.exists(args.model) or not os.path.exists(args.input):
        raise FileNotFoundError("Please make sure the model and input paths are correct!")
    interpreter = Interpreter(args.model)
    # allocate tensors
    interpreter.allocate_tensors()
    # get required input image dimensions
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    print("Image shape must be: (", width, ",", height, ")")
    img = Image.open(args.input)
    image = np.asarray(img).reshape((28, 28, 1))
    time1 = time.time()

    idx, prob = classify(interpreter=interpreter, image=image.copy())

    time2 = time.time()
    time_taken = round(time2-time1)
    print("Classification time: ", time_taken, " Seconds.")
    print(idx, prob)
    return 0


if __name__ == '__main__':
    sys.exit(main())
