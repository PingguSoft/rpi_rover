import platform
import numpy as np
from utils   import *
from components.Mapper import Mapper
from manager import Manager
from components.component import Component
from components.camera import PiCamera
from components.display import Display
from components.VideoWriter import VideoWriter
from components.cv.CVLaneDetection import CVLaneDetection
from components.tf.LanePredictor import LanePredictor
from components.tf.RoadSignDetector import RoadSignDetector
from components.StreamServer import StreamServer
from components.KeyControl import KeyControl

_is_ms_win = platform.system() in ('Windows')
_is_rpi    = platform.machine() in ('armv7l', 'armv6l', 'aarch64')

if _is_rpi:
    from components.actuator import SteeringPWM
    from components.actuator import ThrottlePWM
    from components.RCReceiver import RCReceiverPPM

if _is_ms_win:
    pass
else:
    pass

###############################################################################
# main
###############################################################################
def main():
    np.set_printoptions(formatter={'float_kind': lambda x: "%.4f" % x})
    _rate  = 60
    _debug = False
    _log   = init_log(log_level=logging.INFO)
    _log.info('>> drive started !!')

    Component.disable_thread(_debug)
    Component.set_log(_log)

    _manager = Manager(log=_log)
    _manager.add(PiCamera(160, 120, fps=_rate, video=None if _is_rpi else 'drive_test.mp4'))
    _manager.add(KeyControl())

    if _is_rpi:
        _manager.add(RCReceiverPPM(pin=4))
        _manager.add(SteeringPWM(pin_servo=12))
        # _manager.add(ThrottleMotCtrl3Pin(pin_in1=26, pin_in2=19, pin_pwm=13))
        _manager.add(ThrottlePWM(pin_pwm=13))

    _manager.add(VideoWriter('.rec/'))
    _manager.add(CVLaneDetection(rate_hz=_rate, is_show=True), run_condition='stat/pilot_m1')
    _manager.add(LanePredictor(rate_hz=_rate, model_path='./.train_outputs/rover_pi2_final_q8.tflite'), run_condition='stat/pilot_m2')
    _manager.add(RoadSignDetector(rate_hz=_rate, model_path='./.train_outputs/road_signs_q8.tflite'), run_condition='stat/pilot_m2')
    # _manager.add(StreamServer(rate_hz=_rate))
    _manager.add(Mapper(angle_limit=35, throttle_limit=100))
    _manager.add(Display(scale=2, mem=_manager.get_mem(), rate_hz=_rate, key_wait=_debug))
    _manager.start(rate_hz=_rate, verbose=True)

if __name__ == '__main__':
    main()
