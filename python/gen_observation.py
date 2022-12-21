import numpy as np
from scipy.signal import fftconvolve

from MPM.python import make_sphere, kernel, global_variables as gv, observation3D
from MPM.python.prox import utils as utils


def gen_observation(kernel_mu, kernel_C, sigma_noise=0.2):
    my_sphere = make_sphere.make_sphere()
    # print(my_sphere.shape)
    generated_h = kernel.gaussian_kernel(kernel_C, kernel_mu)
    im = gv.a_sim + gv.b_sim * fftconvolve(my_sphere, generated_h, 'same')
    observation3D.observ(my_sphere, 0, "Grosse bille")
    observation3D.observ(generated_h, kernel_mu[0], "Noyau généré")
    observation3D.observ(im, kernel_mu[0], "Bille convoluée non bruitée")
    im += np.random.randn(im.shape[0], im.shape[1], im.shape[2]) * sigma_noise
    observation3D.observ(im, kernel_mu[0], "Bille convoluée bruitée")
    observation3D.observ(my_sphere, 0, "mysphere")
    observation3D.observ(im, 0, "mysphere")
    return im, generated_h, my_sphere