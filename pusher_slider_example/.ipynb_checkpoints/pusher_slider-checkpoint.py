import time
import numpy as np
import math
# import gym
# from gym.spaces import Box
import pybullet as bullet_client
import pybullet_data as pd
import os
from copy import deepcopy

dir_path = os.path.dirname(os.path.realpath(__file__))

def draw_coordinate(id, **kwargs):
    bullet_client.addUserDebugLine([0,0,0],[0.1,0,0],[1,0,0],parentObjectUniqueId=id, lineWidth=5, **kwargs)
    bullet_client.addUserDebugLine([0,0,0],[0,0.1,0],[0,1,0],parentObjectUniqueId=id, lineWidth=5, **kwargs)
    bullet_client.addUserDebugLine([0,0,0],[0,0,0.1],[0,0,1],parentObjectUniqueId=id, lineWidth=5, **kwargs)

class PusherSlider(object):
    def __init__(self, render=False, time_step = 1./200., frame_skip=1):
        self._time_step = time_step
        self._frame_skip = frame_skip
        self._render = render
        if render:
            bullet_client.connect(bullet_client.GUI)
            bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_GUI, 0)
        else:
            bullet_client.connect(bullet_client.DIRECT)
        bullet_client.setAdditionalSearchPath(pd.getDataPath())
        bullet_client.setTimeStep(time_step)
        bullet_client.setGravity(0., 0., -9.81)
        flags = bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        
        # load up the ground
        bullet_client.loadURDF("plane.urdf", np.array([0.,0.,0.]), flags=flags, useFixedBase=True)
        # load up the pusher and slider
        self.pusher_id = bullet_client.loadURDF(dir_path+'/urdf/panda.urdf',
                                        np.array([-0.3,0.2,0]), useFixedBase=True, flags=flags)
        self.slider_id = bullet_client.loadURDF(dir_path+'/urdf/slider.urdf',
                                        np.array([0.,0.,0.1]), flags=flags)
