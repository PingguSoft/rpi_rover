import time
import pigpio
from utils                import *
from components.component import Component

class SteeringPWM(Component):
    def __init__(self, pin_servo, servo_comp=0):
        Component.__init__(self)
        self.pigpio = pigpio.pi()
        self.pin_servo = pin_servo
        self.comp_servo_pwm = servo_comp
        self.min_servo_pwm = 500
        self.max_servo_pwm = 2500
        self.int_servo_pwm = self.max_servo_pwm - self.min_servo_pwm
        self.oldAngle = 0
        self.pwmServoHz = 50

    def inputs(self):
        return ['angle']

    def outputs(self):
        return []

    def start(self):
        self.pigpio.set_mode(self.pin_servo, pigpio.OUTPUT)
        self.pigpio.set_PWM_frequency(self.pin_servo, self.pwmServoHz)
        self.pigpio.set_PWM_range(self.pin_servo, 4095)
        self.pigpio.set_PWM_dutycycle(self.pin_servo, 0)

    # angle : -1 ~ 1 (-90 degree ~ 90 degree)
    def update(self, angle):
        if angle is None:
            return

        angle = 90 + (angle * 90)
        us    = int(self.min_servo_pwm + (self.int_servo_pwm * angle / 180) + self.comp_servo_pwm)
        duty  = int((4095 * self.pwmServoHz) / 1000000 * us)
        # self.log().info(f'angle={angle:.2f}, us={us:4d}, duty={duty:4d}')
        if angle != self.oldAngle:
            self.pigpio.set_PWM_dutycycle(self.pin_servo, duty)
            self.oldAngle = angle

    def stop(self):
        self.pigpio.write(self.pin_servo, 0)
        self.pigpio.stop()
        self.pigpio = None


class ThrottlePWM(Component):
    def __init__(self, pin_pwm):
        Component.__init__(self)
        self.pigpio = pigpio.pi()
        self.pin_pwm = pin_pwm
        self.min_pwm = 1000
        self.max_pwm = 2000
        self.mid_pwm = (self.min_pwm + self.max_pwm) // 2
        self.int_pwm = self.max_pwm - self.min_pwm
        self.oldAngle = 0
        self.pwmHz    = 50

    def inputs(self):
        return ['throttle']

    def outputs(self):
        return []

    def start(self):
        self.pigpio.set_mode(self.pin_pwm, pigpio.OUTPUT)
        self.pigpio.set_PWM_frequency(self.pin_pwm, self.pwmHz)
        self.pigpio.set_PWM_range(self.pin_pwm, 4095)

        self.log().info("Init ESC")
        self.update(1.0)
        time.sleep(0.01)
        self.update(-1.0)
        time.sleep(0.01)
        self.update(0)
        time.sleep(1)


    # throttle : -1 ~ 1
    def update(self, throttle):
        if throttle is None:
            return

        us    = self.mid_pwm + (self.int_pwm / 2 * throttle)
        duty  = int((4095 * self.pwmHz) / 1000000 * us)
        self.log().debug(f'throttle={throttle:4.1f}, us={us:5.1f}, duty{duty:5d}')

        if throttle != self.oldAngle:
            self.pigpio.set_PWM_dutycycle(self.pin_pwm, duty)
            self.oldAngle = throttle

    def stop(self):
        self.pigpio.write(self.pin_pwm, 0)
        self.pigpio.stop()
        self.pigpio = None



class ThrottleMotCtrl3Pin(Component):
    def __init__(self, pin_in1, pin_in2, pin_pwm, factor=1.0):
        Component.__init__(self)
        self.pigpio = pigpio.pi()
        self.pin_in1 = pin_in1
        self.pin_in2 = pin_in2
        self.pin_pwm = pin_pwm
        self.factor  = factor
        self.old_duty = 0
        self.pigpio.set_mode(self.pin_in1, pigpio.OUTPUT)
        self.pigpio.set_mode(self.pin_in2, pigpio.OUTPUT)
        self.pigpio.write(self.pin_in1, 0)
        self.pigpio.write(self.pin_in2, 0)
        self.pwmSpeedHz = 400
        self.pigpio.set_mode(self.pin_pwm, pigpio.OUTPUT)
        self.pigpio.set_PWM_frequency(self.pin_pwm, self.pwmSpeedHz)
        self.pigpio.set_PWM_range(self.pin_pwm, 255)
        self.pigpio.set_PWM_dutycycle(self.pin_pwm, 0)

    def inputs(self):
        return ['throttle']

    def outputs(self):
        return []

    # throttle : -1 ~ 1 (pwm : -255 ~ 255)
    def update(self, throttle):
        if throttle is None:
            return
        if throttle == 0:
            self.pigpio.write(self.pin_in1, 0)
            self.pigpio.write(self.pin_in2, 0)
        elif throttle > 0:
            self.pigpio.write(self.pin_in1, 1)
            self.pigpio.write(self.pin_in2, 0)
        else:
            self.pigpio.write(self.pin_in1, 0)
            self.pigpio.write(self.pin_in2, 1)

        duty = int(abs(throttle) * self.factor * 255)
        if throttle != self.old_duty:
            self.pigpio.set_PWM_dutycycle(self.pin_pwm, duty)
            self.old_duty = duty

    def stop(self):
        self.pigpio.write(self.pin_in1, 0)
        self.pigpio.write(self.pin_in2, 0)
        self.pigpio.write(self.pin_pwm, 0)
        self.pigpio.stop()
        self.pigpio = None


if __name__ == '__main__':
    log      = init_log(log_level=logging.DEBUG)
    steer    = SteeringPWM(12)
    steer.start()
    # throttle = ThrottleMotCtrl3Pin(26, 19, 13)
    throttle = ThrottlePWM(13)
    throttle.start()

    angle = 0
    speed = 0
    angle_step = 1
    speed_step = 0.02
    ctr   = 0

    while ctr < 2:
        log.debug(f'angle={angle:3d}, speed={speed:5.2f}')
        steer.update(angle / 90)
        throttle.update(speed)
        speed = speed + speed_step
        angle = angle + angle_step
        if angle >= 40 or angle <= -40:
            angle_step = -angle_step
            #ctr += 1
        if speed >= 1.0 or speed <= -1.0:
            speed_step = -speed_step
            ctr += 1

        time.sleep(0.02)

    steer.update(0)
    time.sleep(0.5)
    steer.stop()
    throttle.update(0)
    time.sleep(0.5)
    throttle.stop()
