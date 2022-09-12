import numpy as np
from .proxfk import myprint


def c(D, x, mu, eps):
    Q = 3
    myprint("creating c_n : ", D, mu, eps)
    return Q / 2 * np.log(2 * np.pi) + 1 / 2 * miniphi(D, eps) + 1 / 2 * np.einsum('hijk, kl,hijl-> hij', (x - mu),
                                                                                   (D + eps * np.eye(3)), (x - mu))


def miniphi(D, eps):
    if check_symmetric(D):
        return np.log(np.linalg.det(D + eps * np.eye(3)))
    else:
        raise Exception('D isn\'t symmetric')


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
