import numpy as np
import random
import sys
import pickle as pkl
import glob
import matplotlib.pyplot as plt

sys.path.append('../')

from ctrl_lib.basis import psi

def mapData2Func(data, f):
    if f is None:
        def f(x): x
    out_data = []
    for x in data:
        fx = f(x)
        out_data.append(fx)
    return np.stack(out_data)
random.seed(42)
def getRandomUniqueInts(n, start, end):
    arr = []
    tmp = random.randint(start, end)

    for _ in range(n):
        while tmp in arr:
            tmp = random.randint(start, end)
        arr.append(tmp)
    
    return np.array(arr)


fnames = glob.glob('expert_data/*.pkl')

sampled_state = []
sampled_ctrl = []
sampled_next_state = []

for fname in fnames:
    log = pkl.load(open(fname, 'rb'))
    state = np.stack(log['state'])
    ctrl = np.stack(log['u'])
    next_state = np.stack(log['next state'])
    N = state.shape[0]
    # rnd_idxs = np.random.randint(0, N-1, size=(200,))
    print(N)
    rnd_idxs = getRandomUniqueInts(100, 0, N-1)
    _sampled_state = state[rnd_idxs, :]
    _sampled_ctrl = ctrl[rnd_idxs,:]
    _sampled_next_state = next_state[rnd_idxs,:]
    # _sampled_state = state[:, :]
    # _sampled_ctrl = ctrl[:,:]
    # _sampled_next_state = next_state[:,:]

    sampled_state.append(_sampled_state)
    sampled_ctrl.append(_sampled_ctrl)
    sampled_next_state.append(_sampled_next_state)
    # plt.plot(state[:,-1])
    # plt.show()
sampled_state = np.concatenate(sampled_state)
sampled_ctrl = np.concatenate(sampled_ctrl)
sampled_next_state = np.concatenate(sampled_next_state)

sampled_psi = mapData2Func(sampled_state, f=psi)
sampled_next_psi = mapData2Func(sampled_next_state, f=psi)

X = np.concatenate([sampled_psi, sampled_ctrl], axis=1)
Y = sampled_next_psi

n = sampled_psi.shape[1]
m = sampled_ctrl.shape[1]
A = (np.linalg.pinv(X) @ Y).T

np.save('lsq_a.npy', A[:,:n])
np.save('lsq_b.npy', A[:, n:])

print(sampled_next_psi.shape)
print(sampled_psi.shape)
print(sampled_ctrl.shape)

np.save('PsiX.npy', sampled_psi.T)
np.save('PsiY.npy', sampled_next_psi.T)
np.save('U.npy', sampled_ctrl.T)