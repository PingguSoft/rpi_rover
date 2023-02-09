import time
import inspect
import numpy as np
from multiprocessing  import current_process
from memory import Memory
import queue
from components.component import Component
from components.cv.CVLaneHistogram import CVLaneHistogram
from components.cv.CVLaneLKAS import CVLaneLKAS

###############################################################################
#
###############################################################################
class CVLaneDetection(Component):
    def __init__(self, rate_hz, is_show):
        Component.__init__(self)
        self._hz       = rate_hz
        self._roi      = None
        self._algo     = None
        self._pos      = 0
        self._inQ      = queue.Queue(1)
        self._outputs  = None
        self._profiles = [
            {
                'cls' : CVLaneHistogram,
                'roi' : [160, 120, 30, 97, 10, 115], # w, h, top_w, top_h, bot_w, bot_h
                'col' : 4
            },
            {
                'cls' : CVLaneLKAS,
                'roi' : [160, 120, 0, 70, 0, 120],
                'col' : 3
            }
        ]
        sel = 1
        self._roi  = self.get_roi(self._profiles[sel]['roi'])
        self._algo = self._profiles[sel]['cls'](self._log, self._roi, is_show, self._profiles[sel]['col'])

    def inputs(self):
        return ['image/cam']

    def outputs(self):
        return ['pilot/angle', 'pilot/throttle', 'image/cv_lanedet']

    def is_multi_thread(self):
        return True

    # def is_multi_process(self):
    #     return True

    def start(self):
        pass

    def stop(self):
        self._inQ.put(None, block=False)

    def update(self, image):
        if image is None:
            angle    = 0
            throttle = 0
            out_img  = None
        else:
            outputs = self._algo.process(self._pos, image)
            angle, throttle, out_img  = outputs
            self._pos += 1
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

    def thread_update(self, inputs):
        try:
            if self._inQ.full():
                self._inQ.get(block=False)
            self._inQ.put(inputs, block=False)
        except:
            pass

        return self._outputs

    # #
    # # process_run
    # #
    # """
    # with multiprocess, it is run in separated process
    # and start and stop function will not invoked from the manager so they should be called in run_process
    # """
    # def process_run(self, is_running, mem_name):
    #     Component.process_run(self, is_running, mem_name)
    #
    #     self.start()
    #     mem = Memory(mem_name)
    #     self._log.info(f'{current_process().name} {__class__.__name__}.{inspect.currentframe().f_code.co_name}')
    #
    #     while is_running.is_set():
    #         start_time = time.time()
    #
    #         inputs = mem.get(self.inputs())
    #         self.update(*inputs)
    #         mem.put(self.outputs(), [self._angle, self._img_out])
    #
    #         sleep_time = 1.0 / self._hz - (time.time() - start_time)
    #         if sleep_time > 0.0:
    #             time.sleep(sleep_time)
    #         else:
    #             pass
    #             # print a message when could not maintain loop rate.
    #             # self._log.warning('WARN::Vehicle: jitter violation in vehicle loop '
    #             #                     'with {0:4.0f}ms'.format(abs(1000 * sleep_time)))
    #
    #     self.stop()
    #     self._log.info(
    #         f' {__class__.__name__}.{inspect.currentframe().f_code.co_name} FINISHED !!!')
