import numpy as np
import global_variables as gv

def prox(a, b, y, k):
    gam = gv.gam_b
    return (b + (y-a).flatten().T@k.flatten())/(1+gam*np.linalg.norm(k)**2)
