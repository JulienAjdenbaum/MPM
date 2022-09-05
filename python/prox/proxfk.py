import numpy as np
from scipy.signal import convolve
import math

debug = False


def prox(y, k, p, c, gam, lam, alph, nu0):
    # print("p@p.T", np.max(p))
    grad = convolve(convolve(k, p, 'same')-y, p, "same")
    # grad = convolve(k, p@p.T, "same")-y@p
    # print("grad = ", np.linalg.norm(grad))
    forward = k - alph*grad
    # print("forward = ", np.linalg.norm(forward-k))
    # print("diff", np.linalg.norm(k-forward))
    return proxg(forward, c, gam, lam, nu0)


def mylambertw(w):
    myprint("deb lambert", np.max(w), np.min(w))
    myprint("deb lambert", np.max(np.exp(w)), np.min(np.exp(w)))
    v = np.where(w<1e2, np.real(Lambert_W(np.exp(w))), w-np.log(np.maximum(w, 1e-10)))

    # if np.max(np.where(w<1e2, 0, 1)) != 0:
    #     myprint("aaaaaaaaaaaaaaaaaa")
    #     myprint(np.max(w))
    #     myprint(np.sum(np.where(w<1e2, 0, 1)))
    myprint("fin lambert", np.max(v), np.min(v))
    return v


def proxg(k, c, gam, lam, nu0):
    nu = nu_hat(lam, gam, nu0, k, c)
    myprint("nu = ", nu)
    nu = nu[-1]
    # print("lamb proxg", np.max(w(nu, k, c, lam, gam)))
    return np.real(gam*lam*mylambertw(w(nu, k, c, lam, gam)/(lam*gam))), nu


def nu_hat(lamb, gam, nu0, k, c):
    """
    Newton to get the value nuhat
    :param lamb: value of lamda
    :param gam: value of gamma
    :param nu0: initial value of nu
    :param l
    :return: value of \hat \nu
    """
    epsilon = 1e-9
    maxIter = 10000
    nIter = 0
    nu = [1e-5]
    while True:
        myprint(nu)
        nIter += 1
        fi = phi(nu[-1], k, c, lamb, gam)
        dfi = dphi(nu[-1], k, c, lamb, gam)

        if math.isinf(dfi):
            myprint("isinf !!!!!")
            break
        # print(phi(nu[-1], k, c, lamb, gam)/dfi)
        # if dfi != 0:
        newNu = nu[-1] - fi/dfi
        # else: newNu = 100000
        nu.append(newNu)
        # print("nuuuuuuuu", nu[-1])
        if np.abs(newNu) > 1e50 or np.isnan(nu[-1]) or newNu < -1e50:
            newNu = (np.random.rand(1)-0.5) * 10
            nu[-1] = newNu
            # myprint("Nan : reset nu", nu, nIter)
            # print(k)
            myprint("max k, min k", np.max(k), np.min(k))
            myprint("max c, min c", np.max(c), np.min(c))
            raise Exception('nu is nan :(')
        if nIter > maxIter or np.abs(nu[-1]-nu[-2]) < epsilon:
            break
    myprint(nu)
    return nu


def w(nu, k, c, lamb, gam):
    # myprint("w :", np.max(c), np.max(k), nu)
    return -1 - c + (k-nu)/(lamb*gam)


def phi(nu, k, c, lamb, gam):
    myprint("lamb phi", np.max(w(nu, k, c, lamb, gam)))
    return lamb*gam*np.sum(mylambertw(w(nu, k, c, lamb, gam)/(lamb*gam))) - 1


def dphi(nu, k, c, lamb, gam):
    W = mylambertw(w(nu, k, c, lamb, gam)/(lamb*gam))
    myprint("W dphi", np.max(W), "  ", np.min(W))
    myprint("retour dphi", -np.sum(W/(1+W)))
    return -np.sum(W/(1+W))/(lamb*gam)


def Lambert_W(v):
    w = np.ones(v.shape)
    u = np.inf * w

    n_iter = 0
    while np.sum(np.abs((w - u) / np.maximum(w, 1e-5)) > 1e-07) > 0 and n_iter < 100:
        u = np.copy(w)
        e = np.exp(w)
        f = w * e - v
        w = w - f / (e * (w + 1) - f * (w + 2) / (2 * w + 2))
        n_iter += 1
    return w


def myprint(*args):
    if debug:
        for i in args:
            print(i, end=" ")
        print()
