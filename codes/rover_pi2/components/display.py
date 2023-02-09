import time
import cv2
import queue
import numpy as np
from components.component import Component

###############################################################################
#
###############################################################################
class Display(Component):
    def __init__(self, scale, mem, rate_hz=30, key_wait=False):
        Component.__init__(self)
        self._is_recording = False
        self._angle        = 0
        self._throttle     = 0
        self._is_halt      = False
        self._hz           = rate_hz
        self._image        = None
        self._obj_image    = None
        self._key          = 255
        self._switches     = [0.0] * 10
        self._switches[8]  = 0.25 * scale
        self._keyQ         = queue.Queue(1)
        self._is_key_wait  = key_wait
        self._mem          = mem

    def inputs(self):
        return ['image/cam', 'user/angle', 'user/throttle', 'user/switches']

    def outputs(self):
        return ['user/angle', 'user/throttle', 'user/switches']

    def is_multi_thread(self):
        return True

    def start(self):
        pass

    def stop(self):
        pass

    def update(self, image, angle=None, throttle=None, switches=None):
        self._image = image

        if switches is not None:
            self._switches = switches
        if angle is not None:
            self._angle = angle
        if throttle is not None:
            self._throttle = throttle

        if Component.is_thread_disabled():
            if self._image is not None:
                cv2.imshow(__class__.__name__, self._image)
                key = cv2.waitKeyEx(0 if self._is_key_wait else 1)

        try:
            if not Component.is_thread_disabled():
                key = self._keyQ.get(block=False)
                self._keyQ.task_done()

            if key is not None:
                if key == ord('r'):
                    self._switches[0] = 1.0 if self._switches[0] < 0.5 else 0
                elif key == ord('m'):
                    self._switches[2] = -1.0 if (self._switches[2] == 1.0) else (self._switches[2] + 1.0)
                elif key == ord('q'):
                    self._switches[9] = 1

                elif key == ord('j'):
                    if self._angle > -1.0:
                        self._angle -= 0.1
                elif key == ord('l'):
                    if self._angle < 1.0:
                        self._angle += 0.1
                elif key == ord('i'):
                    if self._throttle < 1.0:
                        self._throttle += 0.1
                elif key == ord('k'):
                    if self._throttle > -1.0:
                        self._throttle -= 0.1
                elif ord('1') <= key <= ord('3'):
                    scale = (key - ord('1') + 1)
                    self._switches[8] = 0.25 * scale
        except Exception as e:
            pass

        return self._angle, self._throttle, self._switches

    #
    # thread_run
    #
    def thread_run(self, is_running):
        last_wins = 0

        while is_running.is_set():
            start_time = time.time()

            if self._image is not None:
                scale = self._switches[8] / 0.25
                if scale <= 0:
                    scale = 1

                # overlay images over base image (black color is transparent)
                list_wins = []

                out = self._image.copy()
                keys = list(self._mem.keys())
                for k in keys:
                    if k.startswith('image/'):
                        if k == 'image/cam':
                            continue
                        obj = self._mem.get([k])[0]
                        if obj is not None:
                            if out.shape == obj.shape:
                                mask = np.any(obj != [0, 0, 0], axis=-1)
                                out[mask] = obj[mask]
                            else:
                                list_wins.append(obj)

                wins = len(list_wins)
                if wins > 0:
                    for i, d in enumerate(list_wins):
                        disp = cv2.resize(d, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                        cv2.imshow(f'{__class__.__name__}_{i:02d}', disp)
                else:
                    disp = cv2.resize(out, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    cv2.imshow(f'{__class__.__name__}_00', disp)

                if wins < last_wins and last_wins > 1:
                    self.log().info(f'{wins}, {last_wins}')
                    for i in range(last_wins, wins, -1):
                        self.log().info(f'destroy {__class__.__name__}_{i - 1:02d}')
                        cv2.destroyWindow(f'{__class__.__name__}_{i - 1:02d}')
                last_wins = wins

                key = cv2.waitKeyEx(1)
                if key != -1:
                    self._keyQ.put(key)
                # self.log().info(f'key:{self._key:3d}')

            sleep_time = 1.0 / self._hz - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                pass
                # print a message when could not maintain loop rate.
                # self.log().warning('WARN::Vehicle: jitter violation in vehicle loop '
                #                   'with {0:4.0f}ms'.format(abs(1000 * sleep_time)))


    def thread_update(self, inputs):
        outputs = self.update(*inputs)
        return outputs
