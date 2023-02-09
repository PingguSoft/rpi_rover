import os
import time
import cv2
import queue
import threading
from datetime             import datetime
from components.component import Component

class VideoWriter(Component):
    def __init__(self, path=None):
        Component.__init__(self)
        self._is_rec = False
        self._outVid = None
        self._outDat = None
        self._path   = path
        self._image  = None
        self._angle  = 0
        self._fps    = 30
        self._inQ    = queue.Queue()

    def inputs(self):
        return ['disp/image', 'angle', 'stat/recording']

    def is_multi_thread(self):
        return True

    def start(self):
        try:
            if not os.path.exists(self._path):
                os.makedirs(self._path, exist_ok=True)
        except Exception as e:
            self.log().error(e)

    def stop(self):
        self.stop_rec()

    def update(self, image, angle, is_rec):
        if not self._is_rec and is_rec:
            self._is_rec = is_rec
            now = datetime.now()
            filename = now.strftime("%m%d_%H%M%S")
            if self._path is not None:
                filename = self._path + filename
            self.log().info(f'video record start : {filename}')
            if image is not None:
                h, w, _ = image.shape
            self.start_rec(filename, w, h, self._fps)
        elif self._is_rec and not is_rec:
            self._is_rec = is_rec
            self.stop_rec()
            self.log().info(f'video record stop')

        if self._is_rec:
            if self._outVid != None:
                self._outVid.write(image)
            if self._outDat != None:
                angle = int(angle * 90)
                self._outDat.write(angle.to_bytes(2, byteorder='little', signed=True))

    def start_rec(self, filename, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._outVid = cv2.VideoWriter(filename + '.mp4', fourcc, fps, (width, height))
        self._outDat = open(filename + '.dat', 'wb')

    def stop_rec(self):
        if self._outVid != None:
            self._outVid.release()
            self._outVid = None

        if self._outDat != None:
            self._outDat.close()
            self._outDat = None


    #
    # thread_run
    #
    def thread_run(self, is_running):
        while is_running.is_set():
            start_time = time.time()

            try:
                inputs = self._inQ.get(block=False)
                self.update(*inputs)
                self._inQ.task_done()
            except Exception as e:
                pass

            sleep_time = (1.0 / self._fps) - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            # else:
            #     # print a message when could not maintain loop rate.
            #     self.log().warning('WARN::Vehicle: jitter violation in vehicle loop '
            #                       'with {0:4.0f}ms'.format(abs(1000 * sleep_time)))


    def thread_update(self, inputs):
        self._inQ.put(inputs)

