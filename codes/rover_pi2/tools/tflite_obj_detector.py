import argparse
import os
import io
import time
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import zipfile
import platform

#
# Global variables
#
_INPUT_NORM_MEAN = 127.5
_INPUT_NORM_STD  = 127.5


#
# Settings
#
np.set_printoptions(formatter={'float_kind':lambda x: "%.4f" % x})
if platform.system() in ('Windows'):
    matplotlib.use('qtagg')
    # print(matplotlib.get_backend())


#
#
#
def draw_objects(image, dict_result, labels, threshold, scale):
    h, w, _ = image.shape
    sy, sx  = scale

    """Draws the bounding box and label for each object."""
    for i, box in enumerate(dict_result['boxes']):
        if dict_result['scores'][i] >= threshold:
            ymin, xmin, ymax, xmax = int(box[0] * sy), int(box[1] * sx), int(box[2] * sy), int(box[3] * sx)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(image, '%s(%.2f)' % (labels.get(dict_result['classes'][i], dict_result['classes'][i]), dict_result['scores'][i]),
                (xmin + 10, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            print('%10s, (%4d, %4d, %4d, %4d), %.2f' % (labels.get(dict_result['classes'][i], dict_result['classes'][i]), xmin, ymin, xmax, ymax, dict_result['scores'][i]))
    return image


def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    # print(f"index={index}, tensor_index={output_details['index']}")
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    if np.dtype(output_details["dtype"]).itemsize == 1:
        output_scale, output_zero_point = output_details["quantization"]
        if output_scale != 0:
            tensor = (tensor - output_zero_point) * output_scale
    return tensor


def set_input_tensor(interpreter, image):
    input_details       = interpreter.get_input_details()
    tensor_index        = input_details[0]['index']
    _, height, width, _ = input_details[0]['shape']

    tensor = interpreter.tensor(tensor_index)()[0]
    tensor.fill(0)
    _, _, ch = tensor.shape
    h, w, _  = image.shape
    scale = min(height / h, width / w)
    h, w  = int(h * scale), int(w * scale)

    result = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR) #interpolation=cv2.INTER_CUBIC)
    # result = tf.compat.v1.image.resize(image, (h, w))

    # normalization
    result = (result - _INPUT_NORM_MEAN) / _INPUT_NORM_STD

    # quantization for np.uint8 or np.int8 input
    if np.dtype(input_details[0]["dtype"]).itemsize == 1:
        input_scale, input_zero_point = input_details[0]["quantization"]
        if input_scale != 0:
            result = result / input_scale + input_zero_point

    result = np.expand_dims(result, axis=0).astype(input_details[0]["dtype"])
    tensor[:h, :w] = np.reshape(result, (h, w, ch))

    return (height / scale, width / scale)


def read_labels(model):
    labels = {}
    with open(model, "rb") as f:
        try:
            with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
                print(zf.namelist())
                for f in zf.namelist():
                    buffer = zf.read(f)
                    words = buffer.decode('utf-8').split('\n')
                    for i, word in enumerate(words):
                        labels[i] = word
        except zipfile.BadZipFile:
            pass

    return labels


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True, help='File path of .tflite file')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Score threshold for detected objects')
    args = parser.parse_args()

    labels = read_labels(args.model)
    split_ext = os.path.splitext(args.model)
    if split_ext[0].endswith('_edgetpu'):
        from pycoral.utils.edgetpu import make_interpreter
        from pycoral.pybind._pywrap_coral import SetVerbosity as set_verbosity
        # set_verbosity(10)
        # interpreter = make_interpreter(args.model)
        interpreter = make_interpreter(*args.model.split('@'))
    else:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=args.model)

    interpreter.allocate_tensors()
    print('\n\n\n')
    print(f'input_details={interpreter.get_input_details()}')
    print('\n')
    print(f'output_details={interpreter.get_output_details()}')
    print('\n')

    frameWidth, frameHeight = 640, 480
    cap = cv2.VideoCapture()
    cap.open(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('cam is not opened')
        return

    print('cam is opened')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

    while True:
        success, image = cap.read()
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start = time.perf_counter()
        scale = set_input_tensor(interpreter, image)
        interpreter.invoke()

        if get_output_tensor(interpreter, 3).size == 1:
            output_desc = {'scores': 2, 'boxes': 0, 'count': 3, 'classes': 1, }
        else:
            output_desc = {'scores': 0, 'boxes': 1, 'count': 2, 'classes': 3, }

        scores = get_output_tensor(interpreter, output_desc['scores']) if 'scores' in output_desc.keys() else None
        boxes = get_output_tensor(interpreter, output_desc['boxes']) if 'boxes' in output_desc.keys() else None
        count = int(get_output_tensor(interpreter, output_desc['count'])) if 'count' in output_desc.keys() else None
        classes = np.int32(get_output_tensor(interpreter, output_desc['classes'])) if 'classes' in output_desc.keys() else None
        if classes is not None and 'classes_offset' in output_desc.keys():
            classes = classes + output_desc['classes_offset']
        dict_result = {'boxes': boxes, 'classes': classes, 'scores': scores, 'count': count}
        inference_time = time.perf_counter() - start
        print('%.2f ms' % (inference_time * 1000))

        output = draw_objects(image, dict_result, labels, args.threshold, scale)
        cv2.imshow("output", output)
        if (cv2.waitKey(1) & 0xff) == ord('q'):
            break

if __name__ == '__main__':
    main()
