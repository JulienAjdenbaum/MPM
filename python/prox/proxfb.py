import numpy as np
import global_variables as gv


def prox(a, b, y, convo):
    gam = gv.gam_b
    return (b + gam * np.sum(np.multiply(y, convo) - a * convo)) / (1 + gam * np.sum(convo ** 2))
