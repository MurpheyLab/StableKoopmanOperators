import numpy as np

def psi(s):
    x,y,th, px, p_pvx, p_pvy = s
    sth = np.sin(th)
    cth = np.cos(th)
    return np.array([
        x, y, th, 
        px, p_pvy, p_pvx,
        sth*p_pvy, cth*p_pvy, sth*p_pvx, cth*p_pvx,
        px*p_pvy, p_pvy*p_pvx 
    ])