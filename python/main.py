import numpy as np
from scipy.signal import fftconvolve
import make_sphere
import kernel
import observation3D

import prox.proxfa as proxfa
import prox.proxfb as proxfb
import prox.proxfh as proxfh
import prox.proxfmu as proxfmu
import prox.proxfd as proxfd
import prox.utils as utils
import time
import matplotlib.pyplot as plt
from matplotlib import rc
# from skimage import io as skio
import global_variables as gv
from scipy.fft import fftn

rc('text', usetex=True)


# np.random.seed(0)

def gen_observation(kernel_mu, kernel_D, sigma_noise = 0.2):
    my_sphere = make_sphere.make_sphere()
    generated_h = kernel.gaussian_kernel(kernel_D, kernel_mu)
    print("mysphere", np.min(my_sphere), np.max(my_sphere))
    print("generated_h", np.min(generated_h), np.max(generated_h), np.sum(generated_h))
    im = gv.a_sim + gv.b_sim * fftconvolve(my_sphere, generated_h, 'same')
    observation3D.observ(my_sphere, 0, "Grosse bille")
    observation3D.observ(generated_h, kernel_mu[0], "Noyau généré")
    observation3D.observ(im, kernel_mu[0], "Bille convoluée non bruitée")
    print("min im", np.min(im), np.max(im))
    im += np.random.randn(im.shape[0], im.shape[1], im.shape[2]) * sigma_noise
    print("min im", np.min(im), np.max(im))
    # if plot_observation:
    #     observation3D.observ(im, kernel_mu[0], "Bille convoluée bruitée")
    #     observation3D.observ(my_sphere)
    #     observation3D.observ(im)
    return utils.saturation(im), generated_h, my_sphere


def from_bille(p, Y):
    x, y, z, XYZ = utils.mymgrid()
    print('x', XYZ.shape)
    # Y, ktrue, p = gen_observation(kernel_size, mutrue, Dtrue, plot)
    D = np.eye(3)
    fftn2 = np.max(np.abs(fftn(p))) ** 2
    gv.gam_h = 2 / fftn2
    # h = kernel.gaussian_kernel(np.diag([5, 1, 2]), [2, 3, 1])
    h = np.zeros(gv.kernel_size)
    # gv.gam_h = 1 / (2 * (np.sum(p)) ** 2)
    # h = np.zeros(gv.kernel_size)
    mu = [0, 0, 0]
    a = 0
    b = 1
    # if gv.plot:
    #     observation3D.observ(h, 0, "k0")

    epsD = 1e-12
    t = time.time()
    # if plot:
    #     observation3D.observ_distri(Y, (1.5, 0.5, 0.5), "Bille observée")
    i_list = []
    norms_new_list = []
    stop = 1000000
    gv.reussi = False


    for i in range(gv.n_iter):
        convo = fftconvolve(h, p, "same")
        newa = proxfa.prox(a, b, Y, convo)
        newb = proxfb.prox(newa, b, Y, convo)
        gv.gam_h = 2 / (b**2 * fftn2)
        newh = proxfh.prox(Y, h, p, convo, newa, newb, D, XYZ, mu, epsD)
        newmu = proxfmu.prox(XYZ, newh, D, mu, gv.lam, epsD)
        newD = proxfd.prox(D, newh, XYZ, newmu, epsD, gv.lam)
        stop_new = np.linalg.norm(h - newh) \
                + np.linalg.norm(mu - newmu)\
                + np.linalg.norm(D - newD) \
                + np.linalg.norm(a - newa)\
                + np.linalg.norm(b - newb)
        stop2 = np.abs(stop-stop_new)/stop_new + 1
        stop = stop_new

        if i % gv.print_n_iter == 0:
            # print("iteration : ", i, "     ", np.linalg.norm(h-newh), "     ",
            # np.linalg.norm(mu-newmu), "     ", np.linalg.norm(D-newD))
            print("\niteration : ", i, 'image :', gv.im_name, 'stop : ', stop,
                  "\nstop2  ", stop2,
                  "\nlambda  ", gv.lam,
                  "\ngam_h  ", gv.gam_h,
                  "\n h-newh", np.linalg.norm(h - newh),
                  "\n mu-newmu", np.linalg.norm(mu - newmu),
                  "\n D-newD", np.linalg.norm(D - newD),
                  "\n a-newa", np.linalg.norm(a - newa),
                  "\n b-newb", np.linalg.norm(b - newb),
                  "\n max h, min h", np.max(newh), np.min(newh),
                  "\n a est", newa,
                  "\n b est", newb,
                  "\n mu est", newmu)
            # print(np.round(newD, 3))
            print(newD)
        i_list.append(i)
        # norm_list.append(np.linalg.norm(D-Dtrue))
        norms_new_list.append(np.linalg.norm(D - newD))
        if stop < gv.stop_criteria or (stop2 < gv.stop_criteria2 and gv.n_iter>100):
            if stop < gv.stop_criteria :
                gv.reussi = True
            print("last iteration :", i)

            print("\niteration : ", i, 'image :', gv.im_name, 'stop : ', stop, "stop2  ", stop2,
                  "\nlambda  ", gv.lam,
                  "\n h-newh", np.linalg.norm(h - newh),
                  "\n mu-newmu", np.linalg.norm(mu - newmu),
                  "\n D-newD", np.linalg.norm(D - newD),
                  "\n a-newa", np.linalg.norm(a - newa),
                  "\n b-newb", np.linalg.norm(b - newb),
                  "\n max h, min h", np.max(newh), np.min(newh),
                  "\n a est", newa,
                  "\n b est", newb,
                  "\n mu est", newmu)
            # print(np.round(newD, 3))
            print(newD)

            break
        a, b, h, mu, D = newa, newb, newh, newmu, newD
    fig1 = plt.figure()
    plt.plot(i_list, norms_new_list)
    plt.xlabel("Itérations")
    plt.ylabel("norm (D-newD)")
    plt.yscale('log')
    plt.title("Evolution de la convergence sur D")
    gv.plots.append(fig1)
    gv.plot_names.append("Evolution de la convergence sur D")
    if gv.plot:
        plt.show()
    print("exec time : ", time.time() - t)
    print("D est", np.round(D, 3))
    print("mu est", np.round(mu, 2))

    kargs = kernel.gaussian_kernel(D, mu)

    return D, mu, a, b, kargs, h, p
