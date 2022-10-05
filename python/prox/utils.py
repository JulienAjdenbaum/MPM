import numpy as np
from .proxfk import myprint
from scipy.signal import convolve
import global_variables as gv

def c(D, x, mu, eps):
    Q = 3
    myprint("creating c_n : ", D, mu, eps)
    return Q / 2 * np.log(2 * np.pi) + 1 / 2 * miniphi(D, eps) + 1 / 2 * np.einsum('hijk, kl,hijl-> hij', (x - mu),
                                                                                   (D + eps * np.eye(3)), (x - mu))


def miniphi(D, eps):
    # if check_symmetric(D):
    return np.log(np.linalg.det(D + eps * np.eye(3)))
    # else:
    #     raise Exception('D isn\'t symmetric')

#
# def check_symmetric(a, rtol=1e-05, atol=1e-08):
#     return np.allclose(a, a.T, rtol=rtol, atol=atol)


def get_barycentre(im):
    xi, yi, zi = np.mgrid[0:im.shape[0],
                          0:im.shape[1],
                          0:im.shape[2]]
    xg = np.sum(np.multiply(xi, im)) / np.sum(im)
    yg = np.sum(np.multiply(yi, im)) / np.sum(im)
    zg = np.sum(np.multiply(zi, im)) / np.sum(im)
    return xg, yg, zg


def get_a(im):
    size = 2
    kernel = np.ones((size, size, size))/size**3
    im_flou = convolve(im, kernel, "same")
    return np.max(im_flou)


def mymgrid():
    shape = gv.kernel_size
    half_size = gv.kernel_size // 2
    x, y, z = np.mgrid[-half_size[0]: half_size[0] + shape[0] % 2,
              -half_size[1]: half_size[1] + shape[1] % 2,
              -half_size[2]: half_size[2] + shape[2] % 2]
    return x, y, z, np.stack((x, y, z), axis=3)
