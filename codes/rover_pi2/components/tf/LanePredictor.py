import cv2
import time
import inspect
import numpy as np
import queue
import math
from components.component import Component
from components.tf.TFLiteUtils import TFLiteUtils
#import tensorflow as tf


###############################################################################
#
###############################################################################
# _HW = '/device:GPU:0'

def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # -90 ~ -1 degree: turn left
    # 0        degree: going straight
    # 1 ~ 90   degree: turn right
    steering_angle_radian = (steering_angle + 90) / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    cv2.putText(heading_image, f'{steering_angle:5.2f}',
                (heading_image.shape[1] // 2 - 40, heading_image.shape[0] - 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                (255, 0, 255), 2)
    # heading_image = cv2.addWeighted(frame, 0.95, heading_image, 1, 1)
    return heading_image


class LanePredictor(Component):
    def __init__(self, rate_hz, model_path):
        Component.__init__(self)
        self._hz       = rate_hz
        self._roi      = None
        self._pos      = 0
        self._inQ      = queue.Queue(1)
        self._outputs  = None
        self._fps_list  = []
        self._last_ev   = 0
        self._model_path = model_path

    def inputs(self):
        return ['image/cam']

    def outputs(self):
        return ['pilot/angle', 'pilot/throttle', 'image/tf_lanedet']

    def is_multi_thread(self):
        return True

    def start(self):
        # with tf.device(_HW):
        #     self.log().info(tf.config.list_physical_devices())
        #     self._model = load_model(model_path)
        self._tf = TFLiteUtils(model_path=self._model_path)
        # self._interpreter = tf.lite.Interpreter(model_path=self._model_path)
        # self._interpreter.allocate_tensors()
        # self._input_details = self._interpreter.get_input_details()[0]
        # self._output_details = self._interpreter.get_output_details()[0]


    def stop(self):
        self._inQ.put(None, block=False)

    def update(self, image):
        if image is None:
            angle    = 0
            throttle = 0
            out_img  = None
        else:
            start = time.time()

            # image preprocess
            height, _, _ = image.shape
            in_image = image[int(height / 2):, :, :]                # remove top half of the image, as it is not relevant for lane following
            in_image = cv2.cvtColor(in_image, cv2.COLOR_BGR2YUV)    # Nvidia model said it is best to use YUV color space
            in_image = cv2.GaussianBlur(in_image, (3, 3), 0)
            in_image = cv2.resize(in_image, (200, 66))              # input image size (200,66) Nvidia model

            # predict
            self._tf.set_input_tensor(in_image)
            self._tf.invoke()
            angle = self._tf.get_output_tensor(0)

            # fps calculation
            diff = time.time() - start
            fps = int(1 / diff) if diff != 0 else 60

            if len(self._fps_list) > 100:
                self._fps_list.pop(0)
            self._fps_list.append(fps)
            fps_avg = int(sum(self._fps_list) / float(len(self._fps_list)))

            # show fps every sec
            ts = time.time()
            diff = ts - self._last_ev
            if diff > 1:
                self._last_ev = ts
                self._log.info(f'fps:{fps_avg:3d}')

            angle = int(angle + 0.5)
            # self.log().info(f'new steering angle: {angle:4d}')
            self._pos += 1
            throttle = 0
            out_img  = display_heading_line(image, angle)

        return angle / 90, throttle, out_img

    def get_roi(self, info):
        width, height, top_w, top_h, bot_w, bot_h = info
        return np.float32([(top_w, top_h), (width - top_w, top_h),
                           (width - bot_w, bot_h), (bot_w, bot_h)])

    #
    # thread_run
    #
    def thread_run(self, is_running):
        while is_running.is_set():
            inputs = self._inQ.get()
            if inputs is not None:
                self._outputs = self.update(*inputs)
                self._inQ.task_done()
            else:
                pass

    def thread_update(self, inputs):
        try:
            if self._inQ.full():
                self._inQ.get(block=False)
            self._inQ.put(inputs, block=False)
        except:
            pass

        return self._outputs
