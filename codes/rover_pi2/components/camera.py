import time
import traceback
import platform
import cv2
import inspect
import queue
from utils                import *
from memory               import Memory
from components.component import Component

_is_rpi = platform.machine() in ('armv7l', 'armv6l', 'aarch64')

if _is_rpi:
    from picamera2 import Picamera2

###############################################################################
#
###############################################################################
class PiCamera(Component):
    def __init__(self, width=160, height=120, fps=30, video=None):
        Component.__init__(self)
        self._width   = width
        self._height  = height
        self._cap     = None
        self._img_out = None
        self._fps     = fps
        self._video   = video

    def outputs(self):
        return ['image/cam']

    # def is_multi_process(self):
    #     return True

    def is_multi_thread(self):
        return True

    #
    # start
    #
    def start(self):
        if self._video is not None:
            self._cap = cv2.VideoCapture(self._video)
            if self._cap is None:
                self.log().error(f'{self._video} open error !!!')
            else:
                maxFrames = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
                self.log().debug(f'maxFrames  : {maxFrames}')
        elif _is_rpi:
            self._cap = Picamera2()
            if self._cap is None:
                self.log().error('camera open error !!!')
            else:
                cfg = self._cap.create_preview_configuration(main={"format": 'RGB888', "size": (self._width, self._height)})
                self._cap.configure(cfg)
                self._cap.start()
            #
            # old rpi camera
            # self._cap = cv2.VideoCapture()
            # self._cap.open(0)
            # if not self._cap.isOpened():
            #     self._cap.release()
            #     self.log().error('camera open error !!!')
            #     return None
            #
            # self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
            # self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
    #
    # stop
    #
    def stop(self):
        if self._video is not None:
            if self._cap is not None and self._cap.isOpened():
                self._cap.release()
        elif _is_rpi:
            if self._cap is not None:
                self._cap.stop()

    #
    # update
    #
    def update(self):
        if self._cap is not None:
            if self._video is not None:
                ret, self._img_out = self._cap.read()
                if not ret:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, self._img_out = self._cap.read()
            elif _is_rpi:
                self._img_out = self._cap.capture_array()

            if self._img_out is not None:
                if self._img_out.shape[0] != self._height or self._img_out.shape[1] != self._width:
                    self._img_out = cv2.resize(self._img_out, (self._width, self._height))
        # self._img_out = cv2.flip(self._img_out, 1)
        return self._img_out

    #
    # thread_run
    #
    def thread_run(self, is_running):
        while is_running.is_set():
            start_time = time.time()
            self.update()
            sleep_time = 1.0 / self._fps - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                pass
                # print a message when could not maintain loop rate.
                # self.log().warning('WARN::Vehicle: jitter violation in vehicle loop '
                #                   'with {0:4.0f}ms'.format(abs(1000 * sleep_time)))

    def thread_update(self, inputs):
        return self._img_out

    #
    # process_run
    #
    """
    with multiprocess, it is run in separated process
    and start and stop function will not invoked from the manager so they should be called in run_process
    """
    def process_run(self, is_running, mem_name):
        Component.process_run(self, is_running, mem_name)
        self.start()
        mem = Memory(mem_name)
        self.log().info(f'{__class__.__name__}.{inspect.currentframe().f_code.co_name}')

        while is_running.is_set():
            start_time = time.time()

            #inputs = mem.get(self.get_inputs())
            outputs = self.update()
            mem.put(self.outputs(), outputs)

            sleep_time = 1.0 / self._fps - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                # print a message when could not maintain loop rate.
                self.log().warning('WARN::Vehicle: jitter violation in vehicle loop '
                                    'with {0:4.0f}ms'.format(abs(1000 * sleep_time)))

        self.stop()
        self.log().info(
            f' {__class__.__name__}.{inspect.currentframe().f_code.co_name} FINISHED !!!')


if __name__ == '__main__':
    log   = init_log(log_level=logging.INFO)
    cam   = PiCamera(log=log, video=None if _is_rpi else 'sample.mp4')
    cam.start()

    while True:
        try:
            frame = cam.update()
            if frame is not None:
                cv2.imshow('cam', frame)
            key = cv2.waitKey(1) & 0xff
            time.sleep(0.02)
        except KeyboardInterrupt:
            break
        except Exception as e:
            traceback.print_exc()

    cam.stop()
