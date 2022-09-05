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
#np.random.seed(0)

def gen_observation(kernel_size, mu, D, plot):
    sigma_noise = 0.1
    volume_size = 10
    sphere_size = 5
    msphere = make_sphere.make_sphere(sphere_size, kernel_size, False)
    k = kernel.gaussian_kernel(kernel_size, D, mu)
    im = convolve(msphere, k, 'same')
    if plot:
        observation3D.observ(msphere, 0, "sphere")
        observation3D.observ(k, mu[0], "ktrue")
        observation3D.observ(im, mu[0], "im before noise")
    im += np.random.randn(im.shape[0], im.shape[1], im.shape[2]) * sigma_noise
    if plot:
        observation3D.observ(im, mu[0], "im noised")
    #     observation3D.observ(msphere)
    #     observation3D.observ(im)
    return im, k, msphere



def main(lam, plot):
    kernel_size = 10
    C = np.array([1, 1, 5])
    mutrue = np.array([1, 2, -3])
    angle = np.array([np.pi / 4, 0, -np.pi / 6])
    Dtrue = kernel.genD(angle, C)

    x, y, z = np.mgrid[- kernel_size: kernel_size + 1,
              -kernel_size: kernel_size + 1,
              - kernel_size: kernel_size + 1]

    X = np.stack((x, y, z), axis=3)

    Y, ktrue, p = gen_observation(kernel_size, mutrue, Dtrue, plot)

    D = np.eye(3)
    k = kernel.gaussian_kernel(kernel_size, np.diag([1, 1, 1]), [0, 0, 0])
    mu = np.random.rand(3)
    # observation3D.observ(k)

    eps = 1e-9
    epsD = 0.001
    gam = 1
    alpha = 0.01
    nu = 0.1
    t = time.time()

    for i in range(1000):
        c = utils.c(D, X, mu, epsD)
        newk, nu = proxfk.prox(Y, k, p, c, gam, lam, alpha, nu)
        newmu = proxfmu.prox(X, newk, D, mu, gam, lam, epsD)
        newD = proxd.prox(D, newk, X, newmu, epsD, lam, gam)
        if i%30==0:
            # print("iteration : ", i, "     ", np.linalg.norm(k-newk), "     ",
                  # np.linalg.norm(mu-newmu), "     ", np.linalg.norm(D-newD))
            print("\niteration : ", i, "lambda  ", lam,
                  "\n k-ktrue", np.linalg.norm(k - ktrue),
                  "\n k-newk", np.linalg.norm(k - newk),
                  "\n mu-newmu", np.linalg.norm(mu - newmu),
                  "\n D-newD", np.linalg.norm(D - newD),
                  "\n max k, min k", np.max(newk), np.min(newk),
                  "\n mu true, mu est", mutrue, newmu)
        if i % 100 == 0:
            print(np.round(D, 3))
            print()
        # print(np.linalg.norm(k - newk) + np.linalg.norm(mu - newmu) + np.linalg.norm(D - newD))
        #
        if np.linalg.norm(k - newk) + np.linalg.norm(mu - newmu) + np.linalg.norm(D - newD) < 1e-5:
            print("last iteration :", i)
            break
        k, mu, D = newk, newmu, newD

        # print(np.sum(k))
    print("exec time : ", time.time()-t)
    print("D est", np.round(D, 3))
    print("D true", np.round(Dtrue, 3))
    print("mu est", np.round(mu, 2))

    kargs = kernel.gaussian_kernel(kernel_size, D, mu)

    if plot:
        observation3D.observ(ktrue, mutrue[0], "ktrue")
        observation3D.observ(kargs, mutrue[0], "kargs")
        observation3D.observ(k, mutrue[0], "k")
        observation3D.observ(Y, mutrue[0], "Y")
        observation3D.observ(convolve(p, k, "same"), mutrue[0], "im_k")
        observation3D.observ(convolve(p, kargs, "same"), mutrue[0], "im_kargs")
    print()
    return D, mu, kargs, k, np.linalg.norm(k-ktrue), np.linalg.norm(kargs-ktrue)


plot = False
x = []
y1 = []
y2 = []
for _lambda in range(1, 20, 2):
    D, mu, kargs, k, norm1, norm2 = main(_lambda/10, plot)
    print(_lambda/10, norm1, norm2)
    x.append(_lambda)
    y1.append(norm1)
    y2.append(norm2)

#%%
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()
