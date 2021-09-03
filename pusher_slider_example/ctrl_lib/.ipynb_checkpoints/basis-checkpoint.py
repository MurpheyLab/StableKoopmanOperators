import numpy as np

def psi(s):
    x,y,th,py,vp,vn = s
    sth = np.sin(th)
    cth = np.cos(th)
    return np.array([
        x, y, th, 
        py, vn, vp,
        sth*vn, cth*vn, sth*vp, cth*vp,
        py*vn, vp*vn 
    ])