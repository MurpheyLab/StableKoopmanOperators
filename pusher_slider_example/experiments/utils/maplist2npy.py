import numpy as np

def maplist2npy(data, f=None):
    if f is None:
        def f(x): x
    out_data = []
    for x in data:
        fx = f(x)
        out_data.append(fx)
    return np.array(out_data)