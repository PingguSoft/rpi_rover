import os
import io
import cv2
import time
import inspect
import numpy as np
import math
import zipfile
#import tensorflow as tf
#from pycoral.utils.edgetpu import make_interpreter

class TFLiteUtils:
    #
    # Global variables
    #
    _INPUT_NORM_MEAN = 127.5
    _INPUT_NORM_STD  = 127.5
    
    def __init__(self, model_path):
        self._model_path  = model_path
        self._labels = self.read_labels(model_path)
        

        split_ext = os.path.splitext(model_path)
        if split_ext[0].endswith('_edgetpu'):
            # from pycoral.pybind._pywrap_coral import SetVerbosity as set_verbosity
            # set_verbosity(10)
            from pycoral.utils.edgetpu import make_interpreter
            self._interpreter = make_interpreter(model_path)
        else:
            import tensorflow as tf
            self._interpreter = tf.lite.Interpreter(model_path)
            # self._interpreter = make_interpreter(model_path)

        # self._interpreter = tf.lite.Interpreter(model_path)
        self._interpreter.allocate_tensors()
        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        print(self._interpreter.get_output_details())

    def set_input_tensor(self, image):
        is_scale_prop = True
        tensor_index = self._input_details[0]['index']
        _, in_h, in_w, _ = self._input_details[0]['shape']

        tensor = self._interpreter.tensor(tensor_index)()[0]
        tensor.fill(0)
        _, _, ch        = tensor.shape
        img_h, img_w, _ = image.shape

        if is_scale_prop:
            scale = min(in_h / img_h, in_w / img_w)
            h, w  = int(img_h * scale), int(img_w * scale)
        else:
            h, w = in_w, in_h

        result = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        # result = tf.compat.v1.image.resize(image, (h, w))

        # normalization
        result = (result - TFLiteUtils._INPUT_NORM_MEAN) / TFLiteUtils._INPUT_NORM_STD

        # quantization for np.uint8 or np.int8 input
        in_type = self._input_details[0]["dtype"]
        if np.dtype(in_type).itemsize == 1:
            input_scale, input_zero_point = self._input_details[0]["quantization"]
            if input_scale != 0:
                result = (result / input_scale) + input_zero_point

        result = np.expand_dims(result, axis=0).astype(in_type)
        tensor[:h, :w] = np.reshape(result, (h, w, ch))

        if is_scale_prop:
            return (in_h / scale, in_w / scale)
        else:
            return img_h, img_w


    def invoke(self):
        return self._interpreter.invoke()


    def get_output_tensor(self, index):
        output_port_details = self._output_details[index]
        # print(f"index={index}, tensor_index={output_port_details['index']}")
        tensor = np.squeeze(self._interpreter.get_tensor(output_port_details['index']))
        if np.dtype(output_port_details["dtype"]).itemsize == 1:
            output_scale, output_zero_point = output_port_details["quantization"]
            if output_scale != 0:
                tensor = (tensor - output_zero_point) * output_scale
        return tensor


    def get_output_tensor_dim(self, index):
        return self._output_details[index]['shape'].size


    def read_labels(self, model_path):
        labels = {}
        with open(model_path, "rb") as f:
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


    def get_labels(self):
        return self._labels

    def class_to_labels(self, classes):
        labels = []
        for i, c in enumerate(classes):
            labels.append(self._labels[c])
        return labels
