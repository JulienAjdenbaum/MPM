import numpy as np
from scipy.signal import fftconvolve
import kernel

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


def PSFEstimation(p, Y):
    x, y, z, XYZ = utils.mymgrid(gv.kernel_size)
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
    alpha = 0
    beta = 1
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
        newalpha = proxfa.prox(alpha, beta, Y, convo)
        newbeta = proxfb.prox(newalpha, beta, Y, convo)
        gv.gam_h = 2 / (beta ** 2 * fftn2)
        newh = proxfh.prox(Y, h, p, convo, newalpha, newbeta, D, XYZ, mu, epsD)
        newmu = proxfmu.prox(XYZ, newh, D, mu, gv.lam, epsD)
        newD = proxfd.prox(D, newh, XYZ, newmu, epsD, gv.lam)
        stop_new = np.linalg.norm(h - newh) \
                   + np.linalg.norm(mu - newmu) \
                   + np.linalg.norm(D - newD) \
                   + np.linalg.norm(alpha - newalpha) \
                   + np.linalg.norm(beta - newbeta)
        stop2 = np.abs(stop - stop_new) / stop_new + 1
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
                  "\n alpha-newalpha", np.linalg.norm(alpha - newalpha),
                  "\n beta-newbeta", np.linalg.norm(beta - newbeta),
                  "\n max h, min h", np.max(newh), np.min(newh),
                  "\n alpha est", newalpha,
                  "\n beta est", newbeta,
                  "\n mu est", newmu)
            # print(np.round(newD, 3))
            print("D_est", newD)
        i_list.append(i)
        # norm_list.append(np.linalg.norm(D-Dtrue))
        norms_new_list.append(np.linalg.norm(D - newD))
        if stop < gv.stop_criteria or (stop2 < gv.stop_criteria2 and gv.n_iter > 100):
            if stop < gv.stop_criteria:
                gv.reussi = True
            print("last iteration :", i)

            print("\niteration : ", i, 'image :', gv.im_name, 'stop : ', stop, "stop2  ", stop2,
                  "\nlambda  ", gv.lam,
                  "\n h-newh", np.linalg.norm(h - newh),
                  "\n mu-newmu", np.linalg.norm(mu - newmu),
                  "\n D-newD", np.linalg.norm(D - newD),
                  "\n alpha-newalpha", np.linalg.norm(alpha - newalpha),
                  "\n beta-newbeta", np.linalg.norm(beta - newbeta),
                  "\n max h, min h", np.max(newh), np.min(newh),
                  "\n alpha est", newalpha,
                  "\n beta est", newbeta,
                  "\n mu est", newmu)
            # print(np.round(newD, 3))
            print(newD)

            break
        alpha, beta, h, mu, D = newalpha, newbeta, newh, newmu, newD
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

    return D, mu, alpha, beta, kargs, h, p
