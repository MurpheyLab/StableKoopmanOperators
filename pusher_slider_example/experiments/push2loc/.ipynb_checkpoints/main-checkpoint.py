
import numpy as np
import time
from datetime import datetime
import pickle as pkl

import sys
sys.path.append('../../')

from pusher_slider_env import PusherSlider
from pusher_slider_env.utils import mat2euler
from ctrl_lib.lqr import FiniteHorizonLQR, InfHorizonLQR
from ctrl_lib.basis import psi

import scipy.io as sio

# np.random.seed(0)
# np.set_printoptions(precision=2, suppress=True)

class Push2Loc(object):
    def __init__(self, method, operator, goal_idx):
        self._seed = int(np.exp(goal_idx))
        np.random.seed(self._seed)
        def get_rand_pi():
            return np.random.uniform(-np.pi/4, np.pi/4)
        def get_rand_x():
            return np.random.uniform(0, 0.4)
        self.env = PusherSlider(render=False, seed=goal_idx)

        self.time_horizon = 5000
        goals = [np.array([get_rand_x(), 0.4, get_rand_pi(), 0., 0., 0.]) for _ in range(1)]
        #     np.array([0.2, 0.4, 0., 0., 0., 0.]),
        #     np.array([0.3, 0.4, 0., 0., 0., 0.]),
        #     np.array([0.2, 0.4, np.pi/4, 0., 0., 0.]),
        #     np.array([0.4, 0.4, -np.pi/4, 0., 0., 0.]),
        #     np.array([0.3, 0.4, np.pi/4, 0., 0., 0.]),
        #     np.array([0.3, 0.4, -np.pi/4, 0., 0., 0.]),
        #     np.array([0.2, 0.4, np.pi/6, 0., 0., 0.]),
        #     np.array([0.4, 0.4, -np.pi/6, 0., 0., 0.]),
        # ]
        if operator == 'stable':
            A = np.load('sample_amatrix.npy')
            B = np.load('sample_bmatrix.npy')
        elif operator == 'least squares':
            A = np.load('lsq_a.npy')
            B = np.load('lsq_b.npy')
        
        # 50 weight for the theta componenet
#         Q = np.diag([800., 800., 50., 10.] + [0. for i in range(8)])
        Q = np.diag([100., 100., 100., 1.] + [0. for i in range(8)])

        R = np.diag([10.**4, 10.**4])
        self.Q = Q
        self.R = R
        _target_state = goals[0]#goals[goal_idx]
        target_state = psi(_target_state)

        if method == 'finite horizon':
            self.lqr = FiniteHorizonLQR(A, B, Q, R, 
                        horizon=self.time_horizon, target_state=target_state)
        elif method == 'infinite horizon':
            self.lqr = InfHorizonLQR(A, B, Q, R, target_state=target_state)

        self.log = {
            # 'p pos in s'  : [],
            # 'p vel in s'  : [],
            # 's pos in w'  : [],
            # 's vel in w'  : [],
            # 's ori in w'  : [],
            'method' : method,
            'operator' : operator,
            'traj cost' : [],
            'Q' : Q,
            'R' : R,
            'target state' : _target_state,
            'err' : []
        }

    def start_experiment(self):
        # for t in range(self.time_horizon):
        state = self.env.reset()
        for t in range(self.time_horizon):
            # ad = np.array([0.,0.])
            ad = self.lqr(psi(state))
            state, u = self.env.step(ad)
            err = psi(state) - self.lqr.target_state
            cost = np.dot(np.dot(err, self.Q),err) + np.dot(np.dot(u, self.R), u)
            self.log['traj cost'].append(cost)
            self.log['err'].append(err.copy())
            # time.sleep(1./60.)

        print('saving data')
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        pkl.dump(self.log, open('data/'+date_str+'{}'.format(self._seed)+'-log.pkl', 'wb'))          


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--operator', type=str, required=True)
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()

if __name__=='__main__':
    print(args.method, args.operator, args.seed)
    experiment = Push2Loc(args.method, args.operator, args.seed)
    experiment.start_experiment()
