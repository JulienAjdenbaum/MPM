import numpy as np
import global_variables as gv

def prox(a, b, y, k):
    gam = gv.gam_a
    return (a + gam*np.sum(y-b*k))/(1+gam*k.shape[0]*k.shape[1]*k.shape[2])
