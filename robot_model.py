#!/usr/bin/env python3
"""Functions for modeling ROSBot"""

#from math import cos, sin
#import numpy as np


def model_parameters():
    """Returns two constant model parameters"""
    # This is a stub. Write your code here.
    # return param_k, param_d


def system_matrix(state_theta):
    """
    Returns a numpy array with the A(theta) matrix
    for a differential drive robot
    """
    # This is a stub. Write your code here.
    # return system_matrix_A


def system_field(state_z, input_u):
    """Computes the field at a given state for the dynamical model"""
    # This is a stub. Write your code here.
    # return dot_state_z


def euler_step(state_z, input_u, step_size):
    """Integrates the dynamical model for one time step using Euler's method"""
    # This is a stub. Write your code here.
    # return state_z_next


def twist_to_speeds(speed_linear, speed_angular):
    """
    Given the desired linear and angular velocity of the robot, returns
    normalized speeds for the left and right motors. Speeds needs to be
    thresholded to be between -1.0 (backward at maximum speed) and 1.0
    (forward at maximum speed).
    """
    # break down into cases
    if speed_angular == 0: # go straight
        speed_left = speed_linear
        speed_right = speed_linear
    elif speed_angular > 0: # turn left
        speed_left = speed_linear * -0.25 # negative to turn faster
        speed_right = speed_linear
    else: # go right
        speed_left = speed_linear
        speed_right =  speed_linear * -0.25

    speed_right = speed_right + speed_angular
    # normalize speeds as necessary
    max_val = max(abs(speed_left), abs(speed_right), 1.0)
    speed_left = speed_left / max_val
    speed_right = speed_right / max_val
    return speed_left, speed_right



class KeysToVelocities():
    '''
    Class to translate cumulative key strokes to speed commands
    '''

    def __init__(self):
        # initialize attributes here
        # This is a stub. Write your code here.
        self.speed_linear = 0
        self.speed_angular = 0
        self.text_description = 'Initial values'

    def update_speeds(self, key):
        '''function to update motor speeds'''
        # This is a stub. Insert your code here
        return self.speed_linear, self.speed_angular, self.text_description


class StampedMsgRegister():
    '''
    Store a previous message, and compute delay with respect to current one
    '''
    def __init__(self):
        # initialize attributes here
        pass

    def replace_and_compute_delay(self, msg):
        '''
        Compute delay between current and stored message,
        then replace stored message with current one
        '''
        # return time_delay, msg_previous

    def previous_stamp(self):
        '''
        Return stored message
        '''
        # return stamp
