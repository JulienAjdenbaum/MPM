import numpy as np
import MPM.python.global_variables as gv


def prox(a, b, y, convo):
    gam = gv.gam_a
    # print("proooox")
    # print(a)
    # print(b)
    # print(y)
    # print(b*fftconvolve(k, p, "same"))
    # print(np.linalg.norm(y-b*fftconvolve(k, p, "same")))
    # print()

    return (a + gam * np.sum(y - b * convo)) / (1 + gam * convo.shape[0] * convo.shape[1] * convo.shape[2])
