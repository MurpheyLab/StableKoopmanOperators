#!/usr/bin/env python3

import numpy as np
# from stable_koopman_operator import StableKoopmanOperator
from koopman_operator import KoopmanOperator
from quad import Quad
from task import Task, Adjoint
import matplotlib.pyplot as plt
import scipy.linalg
from group_theory import VecTose3, TransToRp, RpToTrans
from lqr import FiniteHorizonLQR
from quatmath import euler2mat

from replay_buffer import ReplayBuffer

import pickle as pkl
from datetime import datetime

import scipy.io as sio
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--T', type=int, default=20)
parser.add_argument('--type', type=int, default=0)

args = parser.parse_args()

np.set_printoptions(precision=4, suppress=True)
# np.random.seed(50) ### set the seed for reproducibility

def get_measurement(x):
    g = x[0:16].reshape((4,4)) ## SE(3) matrix
    R,p = TransToRp(g)
    twist = x[16:]
    grot = np.dot(R, [0., 0., -9.81]) ## gravity vec rel to body frame
    return np.concatenate((grot, twist))

def get_position(x):
    g = x[0:16].reshape((4,4))
    R,p = TransToRp(g)
    return p

def main():
    quad = Quad() ### instantiate a quadcopter
    stable_kop = KoopmanOperator(quad.time_step)
    lqr_kop = KoopmanOperator(quad.time_step)

    Klsq = pkl.load(open('al_k_opt.pkl', 'rb'))[-1].T
    data = sio.loadmat('Quadrotor_ActiveLearning_Stable_Kd.mat', squeeze_me=True)
    Kstable = data['Kd']

    stable_kop.set_operator(Kstable)
    lqr_kop.set_operator(Klsq)
    
    simulation_time = 2000
    default_action = lambda x : np.random.uniform(-0.1, 0.1, size=(4,)) ### in case lqr control returns NAN
    _R = euler2mat(np.random.uniform(-1.,1., size=(3,)))
    _p = np.array([0., 0., 0.])
    _g = RpToTrans(_R, _p).ravel()
    _twist = np.random.uniform(-0.6, +0.6, size=(6,))
    state = np.r_[_g, _twist]

    log = {'lsq': [], 'stable': []}

    m_state = get_measurement(state)
    z0_stab = stable_kop.transform_state(m_state)
    z0_lsq = lqr_kop.transform_state(m_state)
    for t in range(simulation_time):

        #### measure state and transform through koopman observables
        m_state = get_measurement(state)
        ctrl = default_action(None)

        z0_stab = stable_kop.step(z0_stab, ctrl)
        z0_lsq = lqr_kop.step(z0_lsq, ctrl)
        next_state = quad.step(state, ctrl)

        log['lsq'].append(
            np.linalg.norm(z0_lsq-lqr_kop.transform_state(next_state))
        )
        log['stable'].append(
            np.linalg.norm(z0_stab-stable_kop.transform_state(next_state))
        )

        state = next_state 

    print(np.mean(log['lsq']), np.mean(log['stable']))
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")


    path = './prediction_data/'

    if os.path.exists(path) is False:
        os.makedirs(path)
    
    pkl.dump(log, open(path + date_str + '.pkl', 'wb'))

if __name__=='__main__':
    main()
