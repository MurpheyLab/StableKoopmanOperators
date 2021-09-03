
import numpy as np
import time
from datetime import datetime
import pickle as pkl

import sys
sys.path.append('../../')

from pusher_slider_env import PusherSlider
import pusher_slider_env
from pusher_slider_env.utils import mat2euler, euler2quat
from ctrl_lib.lqr import FiniteHorizonLQR, InfHorizonLQR
from ctrl_lib.basis import psi

import pybullet as bullet_client
import scipy.io as sio

# np.random.seed(0)
# np.set_printoptions(precision=2, suppress=True)
def get_rand_pi():
    return np.random.uniform(-np.pi/4, np.pi/4)
def get_rand_x():
    return np.random.uniform(0, 0.4)
class Push2Loc(object):
    def __init__(self):

        self.env = PusherSlider(render=True, seed=None)
        self.env.reset()
        self.time_horizon = 200
        goals = [np.array([get_rand_x(), 0.4, get_rand_pi(), 0., 0., 0.]) for _ in range(20)]

        flags = bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        for goal in goals:
            x,y,pi = goal[:3]
            _id = bullet_client.loadURDF(self.env.dir_path+'/urdf/slider.urdf',
                                            [x,y,0.01], euler2quat([0,0,pi])[[3,0,1,2]],flags=flags)
            bullet_client.changeVisualShape(_id, -1, rgbaColor=[0.8, 0.6, 0.4, 0.2])
        input()
if __name__=='__main__':

    experiment = Push2Loc()
