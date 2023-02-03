import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import make_sphere, kernel, global_variables as gv, observation3D
from prox import utils as utils


def gen_observation(kernel_mu, kernel_C, sigma_noise=0.2, sphere_size=None):
    # print('bbbb', gv.kernel_size)
    my_sphere = make_sphere.make_sphere(size=sphere_size)
    # print(my_sphere.shape)
    # print("line 12")
    generated_h = kernel.gaussian_kernel(kernel_C, kernel_mu)
    # print("line 14")
    # plt.imshow(generated_h[generated_h.shape[0] // 2, :, :])
    # plt.show()
    # plt.imshow(generated_h[:, generated_h.shape[1] // 2, :])
    # plt.show()
    im = gv.a_sim + gv.b_sim * fftconvolve(my_sphere, generated_h, 'same')
    # print("line 16")
    # plt.imshow(im[im.shape[0]//2,:,:])
    # plt.show()
    observation3D.observ(my_sphere, 0, "Grosse bille")
    observation3D.observ(generated_h, kernel_mu[0], "Noyau généré")
    observation3D.observ(im, kernel_mu[0], "Bille convoluée non bruitée")
    im += np.random.randn(im.shape[0], im.shape[1], im.shape[2]) * sigma_noise

    observation3D.observ(im, kernel_mu[0], "Bille convoluée bruitée")
    observation3D.observ(my_sphere, 0, "mysphere")
    observation3D.observ(im, 0, "mysphere")
    return im, generated_h, my_sphere
