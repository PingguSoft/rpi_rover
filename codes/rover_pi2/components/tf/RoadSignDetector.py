import cv2
import time
import queue
import numpy as np
from components.component import Component
from components.tf.TFLiteUtils import TFLiteUtils


###############################################################################
#
###############################################################################
# _HW = '/device:GPU:0'
class RoadSignDetector(Component):
    def __init__(self, rate_hz, model_path):
        Component.__init__(self)
        self._hz       = rate_hz
        self._inQ      = queue.Queue(1)
        self._outputs  = None
        self._last_ev  = 0
        self._fps_list = []
        self._tf = TFLiteUtils(model_path)
        if self._tf.get_output_tensor_dim(3) == 1:
            self._output_desc = {'scores': 2, 'boxes': 0, 'count': 3, 'classes': 1, }
        else:
            self._output_desc = {'scores': 0, 'boxes': 1, 'count': 2, 'classes': 3, }

    def inputs(self):
        return ['image/cam']

    def outputs(self):
        return ['image/objs']

    def is_multi_thread(self):
        return True

    def start(self):
        # with tf.device(_HW):
        #     self.log().info(tf.config.list_physical_devices())
        #     self._model = load_model(model_path)
        pass


    def stop(self):
        self._inQ.put(None, block=False)

    def update(self, image):
        if image is None:
            out_img  = None
        else:
            start    = time.time()
            in_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            scale    = self._tf.set_input_tensor(in_image)

            # for tensorflow
            # with tf.device(_HW):
            #     angle = self._model.predict(X, verbose=0)[0]

            # for tensorflow-lite
            self._tf.invoke()

            scores = self._tf.get_output_tensor(self._output_desc['scores']) if 'scores' in self._output_desc.keys() else None
            boxes = self._tf.get_output_tensor(self._output_desc['boxes']) if 'boxes' in self._output_desc.keys() else None
            count = int(self._tf.get_output_tensor(self._output_desc['count'])) if 'count' in self._output_desc.keys() else None
            classes = np.int32(self._tf.get_output_tensor(self._output_desc['classes'])) if 'classes' in self._output_desc.keys() else None
            if classes is not None and 'classes_offset' in self._output_desc.keys():
                classes = classes + self._output_desc['classes_offset']
            dict_result = {'boxes': boxes, 'classes': classes, 'scores': scores, 'count': count}

            out_img = np.zeros_like(image)
            cnt = self.draw_objects(out_img, dict_result, self._tf.get_labels(), 0.4, scale)
            if cnt == 0:
                out_img = None

            # fps calculation
            fps = int(1 / (time.time() - start))
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
        return out_img


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


    def draw_objects(self, image, dict_result, labels, threshold, scale):
        h, w, _ = image.shape
        sy, sx  = scale
        cnt     = 0
        """Draws the bounding box and label for each object."""
        for i, box in enumerate(dict_result['boxes']):
            if dict_result['scores'][i] >= threshold and all(box >= 0):
                ymin, xmin, ymax, xmax = int(box[0] * sy), int(box[1] * sx), int(box[2] * sy), int(box[3] * sx)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(image, '%s(%.2f)' % (labels.get(dict_result['classes'][i], dict_result['classes'][i]), dict_result['scores'][i]),
                    (xmin + 10, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                self.log().info('%10s, (%4d, %4d, %4d, %4d), %.2f' % (labels.get(dict_result['classes'][i], dict_result['classes'][i]), xmin, ymin, xmax, ymax, dict_result['scores'][i]))
                cnt = cnt + 1
        return cnt