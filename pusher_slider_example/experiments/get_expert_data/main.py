import os
import pprint
import pygame
import numpy as np
import time
from datetime import datetime
import pickle as pkl

import sys
sys.path.append('../../')

from pusher_slider_env import PusherSlider
from pusher_slider_env.utils import mat2euler
from joy_interface import JoyInterface


class TeleopDataCollection(object):
    def __init__(self):

        self.env = PusherSlider()
        self.joy = JoyInterface()

        self.log = {
            'state' : [],
            'next state' : [],
            'u' : []
        }

    def start_experiment(self):
        print('ready whenever you are (presss x)')
        while True:
            self.joy.listen()
            if self.joy.button_data[0]:
                break
        time.sleep(1)
        print('running data collection')
        state = self.env.reset()
        k = 0
        while True:
            self.joy.listen()
            vd = np.array([self.joy.axis_data[1], self.joy.axis_data[0]])
            next_state, u = self.env.step(vd)

            # log the environment data 
            # use copy to avoid weirdness with numpy
            self.log['state'].append(state.copy())
            self.log['next state'].append(next_state.copy())
            self.log['u'].append(u.copy())

            state = next_state.copy()

            time.sleep(1./10.)
            if self.joy.button_data[0] and k > 200:
                break
            k += 1
        print('saving data')
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        pkl.dump(self.log, open('../../data/expert_data/'+date_str+'-log.pkl', 'wb'))          

if __name__=='__main__':

    experiment = TeleopDataCollection()
    experiment.start_experiment()