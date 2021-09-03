import os
import pprint
import pygame
import numpy as np
import time

class JoyInterface(object):
    def __init__(self):

        # pygame requires an annoying amount of inits
        pygame.init()
        pygame.joystick.init()
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

        self.axis_data = {0:0.,1:0.}
        self.button_data = {0:False}
        self.hat_data = {}

    def listen(self):

        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                self.axis_data[event.axis] = round(event.value,2)
            elif event.type == pygame.JOYBUTTONDOWN:
                self.button_data[event.button] = True
            elif event.type == pygame.JOYBUTTONUP:
                self.button_data[event.button] = False
            elif event.type == pygame.JOYHATMOTION:
                self.hat_data[event.hat] = event.value
