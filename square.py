#!/usr/bin/env python3
'''Script to run ROSbot in square'''

from time import sleep
import me416_utilities
from robot_model import twist_to_speeds

left_motor = me416_utilities.MotorSpeedLeft()
right_motor = me416_utilities.MotorSpeedRight()

# slow
square_path = [[0.2, 0.], [0.2, .5], [0.2, 0.], [0.2, .5], [0.2, 0.], [0.2, .5], [0.2, 0.]]
# fast
#square_path = [[1., 0.], [1., 1.], [1., 0.], [1., 1.], [1., 0.], [1., 1.], [1., 0.]]
PATHLENGTH = len(square_path)

for i in range(PATHLENGTH):
    if i%2 == 0:
        # straight paths
        speed_values = twist_to_speeds(square_path[i][0], square_path[i][1])
        left_motor.set_speed(speed_values[0])
        right_motor.set_speed(speed_values[1])
        sleep(2)
        # if fast, decrease time
        # sleep(1)
    else:
        # turning
        speed_values = twist_to_speeds(square_path[i][0], square_path[i][1])
        left_motor.set_speed(speed_values[0])
        right_motor.set_speed(speed_values[1])
        sleep(0.5)
        # if fast, decrease time
        # sleep(0.3)
