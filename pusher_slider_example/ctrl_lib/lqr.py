import numpy as np
from numpy.linalg.linalg import solve
from scipy.linalg import solve_discrete_are

class InfHorizonLQR(object):
    def __init__(self, A, B, Q, R, target_state=None):
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(B.T @ P @ B + R) @ B.T @ P @ A
        self.K = K
        self.target_state = None
        self.target_state = target_state
        self.sat_val = np.inf
    def set_target_state(self, target):
        self.target_state = target
    def __call__(self, state):
        return np.clip(-self.K.dot(state-self.target_state), -self.sat_val, self.sat_val)


class FiniteHorizonLQR(object):
    def __init__(self, A, B, Q, R, horizon=10, target_state=None):

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Rinv = np.linalg.inv(R)

        self.horizon = horizon
        self.sat_val = np.inf#1.0
        self.target_state = None
        self.target_state = target_state
        self.K = self.get_control_gains()
        self.t = 0

    def set_target_state(self, target):
        self.target_state = target

    def get_control_gains(self):
        
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R
        Rinv = self.Rinv
        Pt = self.Q.copy()*0

        self.K = []
        for t in range(self.horizon):
            Kt = -np.linalg.inv(self.R + B.T@Pt@B) @ B.T @ Pt @ A
            Pt = Q + Kt.T @ R @ Kt + (A + B@Kt).T @ Pt @ (A + B@Kt)
            self.K.append(Kt.copy())

        return self.K[::-1]

    def __call__(self, state):
        K = self.K[self.t]
        self.t += 1
        return np.clip(K.dot(state-self.target_state), -self.sat_val, self.sat_val)


class FiniteHorizonTrackingLQR(object):
    def __init__(self, A, B, Q, R, ref, horizon=10, target_state=None):

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Rinv = np.linalg.inv(R)
        self.ref = ref
        self.horizon = horizon
        self.sat_val = np.inf#1.0
        self.K, self.Kv, self.v = self.get_control_gains()
        self.t = 0

    def set_target_state(self, target):
        self.target_state = target

    def get_control_gains(self):
        
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R
        Rinv = self.Rinv
        Pt = self.Q.copy()*0
        vt = np.zeros(self.Q.shape[0])
        self.v = []
        self.K = []
        self.Kv = []
        for t in reversed(range(self.horizon)):
            Kt = -np.linalg.inv(self.R + B.T@Pt@B) @ B.T @ Pt @ A
            Kv = np.linalg.inv(self.R + B.T@Pt@B) @ B.T

            Pt = Q + Kt.T @ R @ Kt + (A + B@Kt).T @ Pt @ (A + B@Kt)
            vt = (A - B@ Kt).T @ vt + np.dot(Q, self.ref(t))
            self.v.append(vt)
            self.Kv.append(Kv)
            self.K.append(Kt.copy())

        return self.K[::-1], self.Kv[::-1], self.v[::-1]

    def __call__(self, state):
        K = self.K[self.t]
        Kv = self.Kv[self.t]
        v = self.v[self.t]
        self.t += 1
        return np.clip(K.dot(state-0*self.target_state) + np.dot(Kv,v), -self.sat_val, self.sat_val)