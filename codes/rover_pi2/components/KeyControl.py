import os
import sys
import platform
from components.component import Component

_is_ms_win = platform.system() in ('Windows')
_is_rpi    = platform.machine() in ('armv7l', 'armv6l', 'aarch64')

if _is_ms_win:
    import msvcrt
else:
    import termios

###############################################################################
#
###############################################################################
class KeyControl(Component):
    def __init__(self):
        Component.__init__(self)
        self._angle        = 0
        self._throttle     = 0
        self._old_settings = 0
        self._switches     = None

    def inputs(self):
        return ['user/angle', 'user/throttle', 'user/switches']

    def outputs(self):
        return ['user/angle', 'user/throttle', 'user/switches']

    def start(self):
        self.init_any_key()

    def stop(self):
        self.term_any_key()

    def update(self, angle=None, throttle=None, switches=None):
        if switches is not None:
            self._switches = switches
        if angle is not None:
            self._angle = angle
        if throttle is not None:
            self._throttle = throttle

        key = self.getch()
        if key != -1:
            if self._switches is None:
                self._switches = [0.0] * 10

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

        return self._angle, self._throttle, self._switches

    #
    # key handling functions
    #
    def init_any_key(self):
        if not _is_ms_win:
            self._old_settings = termios.tcgetattr(sys.stdin)
            new_settings = termios.tcgetattr(sys.stdin)
            new_settings[3] = new_settings[3] & ~(termios.ECHO | termios.ICANON) # lflags
            new_settings[6][termios.VMIN] = 0  # cc
            new_settings[6][termios.VTIME] = 0 # cc
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)

    def term_any_key(self):
        if not _is_ms_win:
            if self._old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def getch(self):
        if _is_ms_win:
            if msvcrt.kbhit():
                return ord(msvcrt.getch())
            return -1
        else:
            ch = os.read(sys.stdin.fileno(), 1)
            if ch and len(ch) > 0:
                return ord(ch)
            return -1