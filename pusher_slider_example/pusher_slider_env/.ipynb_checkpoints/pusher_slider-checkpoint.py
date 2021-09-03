import numpy as np
import pybullet as bullet_client
import pybullet_data as pd
import os

from .utils import (set_joints, draw_coordinate, make_pose, 
                    pose_inv, get_rot_trans_from_pose, mat2euler)

dir_path = os.path.dirname(os.path.realpath(__file__))

class PusherSlider(object):
    def __init__(self, render=True, time_step = 1./100., frame_skip=10):
        self._time_step = time_step
        self._frame_skip = frame_skip
        self._render = render
        if render:
            bullet_client.connect(bullet_client.GUI)
            bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_GUI, 0)
            bullet_client.resetDebugVisualizerCamera(1, 90, -52, [0.,0.,0.])
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
        
        self.slider_pos = np.array([0.2,0.2,0.])
        self.slider_offset = np.array([0.,0.,0.01])
        self.slider_id = bullet_client.loadURDF(dir_path+'/urdf/slider.urdf',
                                        self.slider_pos+self.slider_offset, flags=flags)
        
        # move robot to initial pose
        self.__rest_pose = np.array([0., 0., 0., -2.5, 0., 2.5, 0., 0.02, 0.02])
        set_joints(self.pusher_id, self.__rest_pose)
        draw_coordinate(-1)
        draw_coordinate(self.slider_id)
        
        draw_coordinate(self.pusher_id, parentLinkIndex=12)
        
        self.ee_block_offset = np.array([0., -0.04-0.005, 0.05])
        self.ee_pos = self.slider_pos + self.ee_block_offset
        self.ee_orn = np.array([np.pi,0.,0.])
        self.ee_orn = bullet_client.getQuaternionFromEuler(self.ee_orn)
        # use IK to put robot in exact position
        jnt_pose = bullet_client.calculateInverseKinematics(self.pusher_id, 
                                    12, targetPosition=self.ee_pos, targetOrientation=self.ee_orn,
                                    residualThreshold=1e-5, maxNumIterations=200)
        set_joints(self.pusher_id, jnt_pose)
        ee_state = bullet_client.getLinkState(self.pusher_id, 12)
        
        # TODO: make this an input 
        self._vel_cnst = 0.01
        self._vb = np.array([0.,0.])
        self._alpha = 0.5
    
    @property
    def slider_pose(self):
        slider_pos, slider_ori = bullet_client.getBasePositionAndOrientation(self.slider_id)
        slider_mat  = bullet_client.getMatrixFromQuaternion(slider_ori)
        slider_mat  = np.array(slider_mat).reshape((3,3))
        slider_T = make_pose(slider_pos, slider_mat)
        return slider_T
    
    @property
    def slider_vel(self):
        v, w = bullet_client.getBaseVelocity(self.slider_id)
        return np.concatenate([v[:2], [w[-1]]])

    @property
    def pusher_pose(self):
        ee_state = bullet_client.getLinkState(self.pusher_id, 12, computeLinkVelocity=1)
        ee_pos   = np.array(ee_state[4])
        ee_ori   = np.eye(3)
        ee_T = make_pose(ee_pos, ee_ori)
        return ee_T
    
    def reset(self):
        self.slider_pos[0] = np.random.uniform(0.,0.4)
        self.slider_pos[1] = np.random.uniform(0., 0.6)
        self.ee_pos = self.slider_pos + self.ee_block_offset
        self._vn = np.array([0.,0.])

        bullet_client.resetBasePositionAndOrientation(self.slider_id, 
                                                      self.slider_pos+self.slider_offset,
                                                      np.array([0.,0.,0.,1.]))
        bullet_client.resetBaseVelocity(self.slider_id, np.zeros(3), np.zeros(3))
        
        set_joints(self.pusher_id, self.__rest_pose)
        jnt_pose = bullet_client.calculateInverseKinematics(self.pusher_id, 
                            12, targetPosition=self.ee_pos, targetOrientation=self.ee_orn,
                            residualThreshold=1e-5, maxNumIterations=200)
        set_joints(self.pusher_id, jnt_pose)
    
    def step(self, ab): # pass in velocity in body frame
        pusher_T = self.pusher_pose
        slider_T = self.slider_pose
        pnt_T = pose_inv(slider_T)@pusher_T
        _, pnt_pos = get_rot_trans_from_pose(pnt_T)
        slider_mat, slider_pos = get_rot_trans_from_pose(slider_T)
        _, pusher_pos = get_rot_trans_from_pose(pusher_T)

        vb = self._alpha * self._vb + (1-self._alpha) * ab
        if pnt_pos[0] > 0.04:
            vb[0] = np.clip(vb[0], -np.inf, 0.)
            ab[0] = np.clip(ab[0], -np.inf, 0.)
        if pnt_pos[0] < -0.04:
            vb[0] = np.clip(vb[0], 0., np.inf)
            ab[0] = np.clip(ab[0], 0., np.inf)
        vb[1] = np.clip(vb[1], 0, np.inf)
        ab[1] = np.clip(ab[1], 0, np.inf)
        self._vb = vb

        pusher_vel_d = slider_mat[:2,:2] @ vb 

        pusher_pos_d = pusher_pos + np.concatenate([pusher_vel_d,[0.]]) * self._vel_cnst
        # reset height for good measure
        pusher_pos_d[2] = self.ee_block_offset[2]
        jnt_pose = bullet_client.calculateInverseKinematics(self.pusher_id, 
                            12, targetPosition=pusher_pos_d, targetOrientation=self.ee_orn,
                            residualThreshold=1e-5, maxNumIterations=200)
        for i, jnt in enumerate(jnt_pose):
            bullet_client.setJointMotorControl2(
                            self.pusher_id, i, bullet_client.POSITION_CONTROL, 
                            jnt, force=270.)
        
        # this creates the effective 10Hz control signal
        for _ in range(self._frame_skip):
            bullet_client.stepSimulation()
        
        return pnt_pos[:2], np.array([mat2euler(slider_mat)[-1]]), vb,  \
                slider_pos[:2], self.slider_vel, ab

        
