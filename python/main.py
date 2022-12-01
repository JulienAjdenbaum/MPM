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
    im = gv.a_sim + gv.b_sim*fftconvolve(my_sphere, generated_h, 'same')
    observation3D.observ(my_sphere, 0, "Grosse bille")
    observation3D.observ(generated_h, kernel_mu[0], "Noyau généré")
    observation3D.observ(im, kernel_mu[0], "Bille convoluée non bruitée")
    print("min im", np.min(im))
    im += np.random.randn(im.shape[0], im.shape[1], im.shape[2]) * sigma_noise
    print("min im", np.min(im))
    # if plot_observation:
    #     observation3D.observ(im, kernel_mu[0], "Bille convoluée bruitée")
    #     observation3D.observ(my_sphere)
    #     observation3D.observ(im)
    return im, generated_h, my_sphere


# def main(lam, plot):
#
#
#     x, y, z = np.mgrid[-kernel_size: kernel_size + 1, -kernel_size: kernel_size + 1, -kernel_size: kernel_size + 1]
#
#     X = np.stack((x, y, z), axis=3)
#
#
#
#     D = np.eye(3)
#     k = kernel.gaussian_kernel(kernel_size, np.diag([1, 1, 1]), [0, 0, 0])
#     mu = np.random.rand(3)
#     # observation3D.observ(k)
#
#     epsD = 0.001
#     gam = 1
#     alpha = 0.01
#     t = time.time()
#
#     for i in range(1000):
#         c = utils.c(D, X, mu, epsD)
#         newk = proxfk.prox(Y, k, p, c, gam, lam, alpha)
#         newmu = proxfmu.prox(X, newk, D, mu, gam, lam, epsD)
#         newD = proxd.prox(D, newk, X, newmu, epsD, lam, gam)
#         if i % 30 == 0:
#             # print("iteration : ", i, "     ", np.linalg.norm(k-newk), "     ",
#             # np.linalg.norm(mu-newmu), "     ", np.linalg.norm(D-newD))
#             print("\niteration : ", i, "lambda  ", lam,
#                   "\n k-ktrue", np.linalg.norm(k - ktrue),
#                   "\n k-newk", np.linalg.norm(k - newk),
#                   "\n mu-newmu", np.linalg.norm(mu - newmu),
#                   "\n D-newD", np.linalg.norm(D - newD),
#                   "\n max k, min k", np.max(newk), np.min(newk),
#                   "\n mu true, mu est", mutrue, newmu)
#         if i % 100 == 0:
#             print(np.round(D, 3))
#             print()
#         # print(np.linalg.norm(k - newk) + np.linalg.norm(mu - newmu) + np.linalg.norm(D - newD))
#         #
#         if np.linalg.norm(k - newk) + np.linalg.norm(mu - newmu) + np.linalg.norm(D - newD) < 1e-6:
#             print("last iteration :", i)
#             break
#         k, mu, D = newk, newmu, newD
#
#         # print(np.sum(k))
#     print("exec time : ", time.time() - t)
#     print("D est", np.round(D, 3))
#     print("D true", np.round(Dtrue, 3))
#     print("mu est", np.round(mu, 2))
#
#     kargs = kernel.gaussian_kernel(kernel_size, D, mu)
#
#     if plot:
#         observation3D.observ(ktrue, mutrue[0], "ktrue")
#         observation3D.observ(kargs, mutrue[0], "Estimation du Noyau")
#         observation3D.observ(k, mutrue[0], "Noyau estimé à partir des paramètres")
#         # observation3D.observ(Y, mutrue[0], "Y")
#         observation3D.observ(fftconvolve(p, k, "same"), mutrue[0], "im_k")
#         observation3D.observ(fftconvolve(p, kargs, "same"), mutrue[0], "im_kargs")
#     print()
#     return D, mu, kargs, k, np.linalg.norm(k - ktrue), np.linalg.norm(kargs - ktrue)
# X_plot = []
# y1 = []
# y2 = []
# for _lambda in range(1, 20, 2):
#     chosen_D, chosen_mu, k_args, chosen_k, norm1, norm2 = main(_lambda/10, plot=False)
#     print(_lambda/10, norm1, norm2)
#     X_plot.append(_lambda/10)
#     y1.append(norm1)
#     y2.append(norm2)
#
# plt.plot(X_plot, y1)
# plt.plot(X_plot, y2)
# plt.show()

# path = '/home/julin/Documents/imbilles/4um/'
# imname = '1_max_810_575-630_4um_Pmax_500V_3X_10_0.497umx_0.5z.tif'
# im = skio.imread(path + imname)
# print(im.shape)
# observation3D.observ(im, 0, "bille")


def from_bille(p, Y):
    x, y, z, XYZ = utils.mymgrid()
    print('x', XYZ.shape)
    # Y, ktrue, p = gen_observation(kernel_size, mutrue, Dtrue, plot)
    gv.alpha = 1/(2*np.max(np.abs(fftn(p)))**2)
    gv.gam_h = gv.alpha
    print("valeur de alpha : ", gv.alpha)
    D = np.eye(3)
    # h = kernel.gaussian_kernel(np.diag([1, 1, 1]), [0, 0, 0])
    h = np.zeros(gv.kernel_size)
    mu = [0, 0, 0]
    a = 0
    b = 1
    # if gv.plot:
    #     observation3D.observ(h, 0, "k0")

    epsD = 1e-8
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
        newh = proxfh.prox(Y, h, p, convo, newa, newb, D, XYZ, mu, epsD)
        newmu = proxfmu.prox(XYZ, newh, D, mu, gv._lambda, epsD)
        newD = proxfd.prox(D, newh, XYZ, newmu, epsD, gv._lambda)
        stop_new = np.linalg.norm(h - newh) \
                + np.linalg.norm(mu - newmu)\
                + np.linalg.norm(D - newD) \
                + np.linalg.norm(a - newa)\
                + np.linalg.norm(b - newb)
        stop2 = np.abs(stop-stop_new)/stop_new
        stop = stop_new

        if i % gv.print_n_iter == 0:
            # print("iteration : ", i, "     ", np.linalg.norm(h-newh), "     ",
            # np.linalg.norm(mu-newmu), "     ", np.linalg.norm(D-newD))
            print("\niteration : ", i, 'image :', gv.im_name, 'stop : ', stop,
                  "\nstop2  ", stop2,
                  "\nlambda  ", gv._lambda,
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
                  "\nlambda  ", gv._lambda,
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
