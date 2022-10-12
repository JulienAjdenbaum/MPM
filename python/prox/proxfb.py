import numpy as np
import global_variables as gv
from scipy.signal import fftconvolve

def prox(a, b, y, convo):
    gam = gv.gam_b
    return (b + gam*np.sum(np.multiply(y, convo)-a*convo))/(1+gam*np.sum(convo**2))
    # return b-gam*np.linalg.norm(fftconvolve(y-a-b*fftconvolve(k, p, "same"), fftconvolve(k, p, "same")[::-1, ::-1, ::-1], "same"))
