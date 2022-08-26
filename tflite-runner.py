import os.path

from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np
import time
import argparse
import sys
from tflite_runtime.interpreter import Interpreter
import numpy as np
from PIL import Image


def parse_args(args=[]):
    parser = argparse.ArgumentParser(description='Capstone: Distributed Edge Inference TFLite Converter')
    parser.add_argument("-m", "--model", help="Absolute path to model .tflite file.")
    parser.add_argument("-i", "--input", help="input file to classify")
    parser.add_argument("-l", "--labels", help="Path to labels.txt")
    parser.add_argument("-t", "--topk", default=1, help="Top k element")
    return parser.parse_args(args)


def classify(model_path, input_path, top=1):
    # Create model interpreter
    interpreter = Interpreter(model_path)
    # allocate tensor to get input/output details
    interpreter.allocate_tensors()

    # Get input details
    input_details = interpreter.get_input_details()
    # Get output details
    output_details = interpreter.get_output_details()

    # get input image shape to reshape input if not in the correct shape
    input_shape = input_details[0]['shape']

    # load image data
    img = Image.open(input_path)

    # convert image to numpy array
    data = np.asarray(img, dtype='float32')

    # normalize...
    data = data.reshape(input_shape) / 255.

    # set data as input for our model
    interpreter.set_tensor(input_details[0]['index'], data)

    # get start time
    start = time.time()

    # run the inference
    interpreter.invoke()

    # get end time
    end = time.time()

    # get results...
    res = interpreter.get_tensor(output_details[0]['index'])

    idx = np.argpartition(res.flatten(), kth=-top)[-top:]

    print(f"Execution time: {end-start}")

    return idx


def main():
    args = parse_args(args=sys.argv[1:])

    if not os.path.exists(args.model) or not os.path.exists(args.input):
        raise FileNotFoundError("Please make sure the model and input paths are correct!")
    result = classify(model_path=args.model, input_path=args.input, top=int(args.topk))
    print(result)
    return 0


if __name__ == '__main__':
    sys.exit(main())
