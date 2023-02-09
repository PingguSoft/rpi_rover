import cv2
import time
from components.component import Component

#
# Mapper class
#
class Mapper(Component):
    def __init__(self, angle_limit=40, throttle_limit=80, idx_rec_sw=0, idx_mode_sw=2):
        Component.__init__(self)
        self.angle_limit = angle_limit / 90
        self.throttle_limit = throttle_limit / 100
        self.idx_rec_sw = idx_rec_sw
        self.idx_mode_sw = idx_mode_sw
        self.pilot_mode = 0
        self.is_rec = False

    def inputs(self):
        return ['user/angle', 'user/throttle', 'user/switches', 'pilot/angle', 'pilot/throttle']

    def outputs(self):
        return ['angle', 'throttle', 'stat/pilot_m1', 'stat/pilot_m2', 'stat/recording', 'HALT']


    def constrain(self, amt, low, high):
        return low if amt < low else (high if amt > high else amt)


    def mapping(self, x, in_min, in_max, out_min, out_max):
        dividend = out_max - out_min
        divisor = in_max - in_min
        delta = x - in_min
        ret = None if divisor == 0 else delta * dividend / divisor + out_min
        return ret


    def update(self, user_angle, user_throttle, switches, pilot_angle, pilot_throttle):
        is_halt = False
        if switches is not None:
            # recording key
            self.is_rec = True if switches[self.idx_rec_sw] > 0.5 else False

            # mode key
            #self.is_pilot = True if switches[self.idx_mode_sw] > 0.5 else False
            self.pilot_mode = 1.0 if (switches[self.idx_mode_sw] > 0.7) else \
                (-1.0 if (switches[self.idx_mode_sw] < -0.7) else 0)

            # halt?
            is_halt = True if switches[9] > 0.5 else False

        is_pilot_m1 = (self.pilot_mode == -1)
        is_pilot_m2 = (self.pilot_mode ==  1)

        if self.pilot_mode != 0:
            angle = pilot_angle # * 1.8 if pilot_angle is not None else pilot_angle
            throttle = user_throttle
        else:
            angle = user_angle
            throttle = user_throttle

        angle = 0 if angle is None else self.constrain(angle, -self.angle_limit, self.angle_limit)
        throttle = 0 if throttle is None else self.mapping(self.constrain(throttle, -1.0, 1.0), -1.0, 1.0,
                                                           -self.throttle_limit, self.throttle_limit)
        # self.log().info(f'angle:{angle:5.2f} throttle:{throttle:5.2f}') #, switches:{switches}')

        return angle, throttle, is_pilot_m1, is_pilot_m2, self.is_rec, is_halt
