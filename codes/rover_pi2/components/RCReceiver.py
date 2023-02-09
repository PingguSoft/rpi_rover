import time
import traceback
import pigpio
from utils                import *
from components.component import Component

class RCReceiverPPM(Component):
    MAX_CH  = 10
    MIN_OUT = -1
    MAX_OUT =  1
    def __init__(self, pin, invert=False, min_pwm=1000, max_pwm=2000, stop_pwm = 1500):
        Component.__init__(self)
        self.is_conn  = False
        self.pigpio   = pigpio.pi()
        self.pin      = pin
        self.invert   = invert
        self.min_pwm  = min_pwm
        self.max_pwm  = max_pwm
        self.stop_pwm = stop_pwm

        # standard variables
        self.cur_ch    = 0
        self.last_tick = 0
        self.pulses    = [stop_pwm] * self.MAX_CH
        self.factor    = (self.MAX_OUT - self.MIN_OUT) / (self.max_pwm - self.min_pwm)
        self.signals   = [0] * self.MAX_CH

        if self.log():
            self.log().info(f'RCReceiverPPM gpio {self.pin} created')

    def inputs(self):
        return ['user/angle', 'user/throttle', 'user/switches']

    def outputs(self):
        return ['user/angle', 'user/throttle', 'user/switches']

    def start(self):
        self.pigpio.set_mode(self.pin, pigpio.INPUT)
        self.cb = self.pigpio.callback(self.pin, pigpio.RISING_EDGE, self.cbf)
        if self.log():
            self.log().info(f'start RCReceiverPPM')

    def cbf(self, gpio, level, tick):
        """ Callback function for pigpio interrupt gpio. Signature is determined
            by pigpiod library. This function is called every time the gpio
            changes self.state as we specified RISING_EDGE.  The pigpio callback library
            sends the user-defined callback function three parameters, which it may or may not use
        :param gpio: gpio to listen for state changes
        :param level: rising/falling edge
        :param tick: # of mu s since boot, 32 bit int
        """
        diff = pigpio.tickDiff(self.last_tick, tick)
        self.last_tick = tick
        if diff > 3000:
            self.cur_ch = 0
        else:
            if self.cur_ch < self.MAX_CH:
                diff = max(diff, self.min_pwm)
                diff = min(diff, self.max_pwm)
                self.pulses[self.cur_ch] = diff
                self.cur_ch += 1

    def update(self, angle=None, throttle=None, switches=None):
        diff = pigpio.tickDiff(self.last_tick, self.pigpio.get_current_tick())
        if diff > (self.max_pwm * 100):
            for i in range(self.MAX_CH):
                self.signals[i] = (self.stop_pwm - self.min_pwm) * self.factor
                if self.invert:
                    self.signals[i] = -self.signals[i] + self.MAX_OUT
                else:
                    self.signals[i] += self.MIN_OUT
            if self.is_conn:
                self.is_conn = False
                if self.log():
                    self.log().warning(f'no rc signal {diff:8d}')
        else:
            if not self.is_conn:
                self.is_conn = True
            log = 'RC '
            for i in range(self.MAX_CH):
                self.signals[i] = (self.pulses[i] - self.min_pwm) * self.factor
                if self.invert:
                    self.signals[i] = -self.signals[i] + self.MAX_OUT
                else:
                    self.signals[i] += self.MIN_OUT
                if self.log():
                    log = log + f'CH{i:1d} {self.signals[i]:5.2f}, '
            self.log().debug(f'{log}')

        #self.log().info(f'angle={angle}, throttle={throttle}, switches={switches} is_con={self.is_conn}')

        if self.is_conn:
            if switches is not None:
                s6, s7, s8, s9 = switches[6], switches[7], switches[8], switches[9]
            else:
                s6, s7, s8, s9 = 0, 0, 0, 0

            return self.signals[0], self.signals[1], [self.signals[2], self.signals[3],
                self.signals[4], self.signals[5], self.signals[6], self.signals[7],
                s6, s7, s8, s9]
        else:
            return angle, throttle, switches

    def stop(self):
        self.cb.cancel()
        self.pigpio.stop()


###############################################################################
# main
###############################################################################
if __name__ == "__main__":
    log   = init_log(log_level=logging.DEBUG)
    Component.set_log(log)
    stick = RCReceiverPPM(4)
    stick.start()

    while True:
        try:
            stick.update()
            time.sleep(0.02)
        except KeyboardInterrupt:
            break
        except Exception as e:
            traceback.print_exc()

    stick.stop()
