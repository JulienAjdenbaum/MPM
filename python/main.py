import numpy as np
from scipy.signal import convolve
import make_sphere
import kernel
import observation3D
import prox.proxfd as proxd
import prox.proxfk as proxfk
import prox.proxfmu as proxfmu
import prox.utils as utils
import time
import matplotlib.pyplot as plt
from matplotlib import rc
# from skimage import io as skio
import global_variables as gv
rc('text', usetex=True)


# np.random.seed(0)

def gen_observation(kernel_size, sphere_size, kernel_mu, kernel_D, a=1, sigma_noise = 0.2):
    my_sphere = make_sphere.make_sphere(sphere_size, kernel_size, a=a)
    generated_k = kernel.gaussian_kernel(kernel_size, kernel_D, kernel_mu)
    im = convolve(my_sphere, generated_k, 'same')
    if gv.plot:
        observation3D.observ(my_sphere, 0, "Grosse bille")
        observation3D.observ(generated_k, kernel_mu[0], "Noyau généré")
        observation3D.observ(im, kernel_mu[0], "Bille convoluée non bruitée")

    im += np.random.randn(im.shape[0], im.shape[1], im.shape[2]) * sigma_noise
    # if plot_observation:
    #     observation3D.observ(im, kernel_mu[0], "Bille convoluée bruitée")
    #     observation3D.observ(my_sphere)
    #     observation3D.observ(im)
    return im, generated_k, my_sphere


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
#         observation3D.observ(convolve(p, k, "same"), mutrue[0], "im_k")
#         observation3D.observ(convolve(p, kargs, "same"), mutrue[0], "im_kargs")
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


def from_bille(lam, p, Y):
    kernel_size = np.array(Y.shape) // 2
    x, y, z = np.mgrid[-kernel_size[0]: kernel_size[0] + Y.shape[0] % 2,
              -kernel_size[1]: kernel_size[1] + Y.shape[1] % 2,
              -kernel_size[2]: kernel_size[2] + Y.shape[2] % 2]
    X = np.stack((x, y, z), axis=3)

    # Y, ktrue, p = gen_observation(kernel_size, mutrue, Dtrue, plot)

    D = np.eye(3)
    k = kernel.gaussian_kernel(kernel_size, np.diag([1, 1, 1]), [0, 0, 0])
    mu = [0, 0, 0]
    # observation3D.observ(k)

    epsD = 1e-8
    gamC = 1
    gammu = 1
    t = time.time()
    # if plot:
    #     observation3D.observ_distri(Y, (1.5, 0.5, 0.5), "Bille observée")
    i_list = []
    norms_new_list = []
    for i in range(3000):
        # print(k.shape, p.shape, Y.shape)
        newk = proxfk.prox(Y, k, p, D, X, mu, epsD)
        newmu = proxfmu.prox(X, newk, D, mu, gammu, lam, epsD)
        newD = proxd.prox(D, newk, X, newmu, epsD, lam, gamC)
        if i % 30 == 0:
            # print("iteration : ", i, "     ", np.linalg.norm(k-newk), "     ",
            # np.linalg.norm(mu-newmu), "     ", np.linalg.norm(D-newD))
            print("\niteration : ", i, "lambda  ", lam,
                  "\n k-newk", np.linalg.norm(k - newk),
                  "\n mu-newmu", np.linalg.norm(mu - newmu),
                  "\n D-newD", np.linalg.norm(D - newD),
                  "\n max k, min k", np.max(newk), np.min(newk),
                  "\n mu est", newmu)
            print(newD)
        i_list.append(i)
        # norm_list.append(np.linalg.norm(D-Dtrue))
        norms_new_list.append(np.linalg.norm(D - newD))
        if np.linalg.norm(k - newk) + np.linalg.norm(mu - newmu) + np.linalg.norm(D - newD) < 1e-6:
            print("last iteration :", i)
            break
        k, mu, D = newk, newmu, newD

    plt.plot(i_list, norms_new_list)
    plt.xlabel("Itérations")
    plt.ylabel("norm (D-newD)")
    plt.yscale('log')
    plt.title("Evolution de la convergence sur D")
    plt.show()
    print("exec time : ", time.time() - t)
    print("D est", np.round(D, 3))
    print("mu est", np.round(mu, 2))

    kargs = kernel.gaussian_kernel(kernel_size, D, mu)

    return D, mu, kargs, k, p
