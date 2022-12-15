import numpy as np
from scipy.signal import fftconvolve
import math
import prox.utils as utils
import global_variables as gv
from scipy.special import lambertw

def prox(y, k, X, convo, a, b, D, x, mu, eps):
    lam = gv.lam
    alph = gv.gam_h
    gam = gv.gam_h

    grad = fftconvolve(b ** 2 * convo + b * a - b * y, X[::-1, ::-1, ::-1], "same") * utils.saturation_der(a + b * convo)

    forward = k - alph * grad
    # print("forward = ", np.linalg.norm(forward-k))
    # print("diff", np.linalg.norm(k-forward))
    c = utils.c(D, x, mu, eps)
    return proxg(forward, c, gam, lam)

# def mylambertw(x):
#     myprint("deb lambert", np.max(x), np.min(x))
#     myprint("deb lambert", np.max(np.exp(x)), np.min(np.exp(x)))
#     v = np.where(x < 1e2, Lambert_W(x), x - np.log(np.maximum(x, 1e-10)))
#     myprint("fin lambert", np.max(v), np.min(v))
#     return v

def w(nu, h, c, lamb, gam):
    # myprint("w :", np.max(c), np.max(k), nu)
    return -1 - c + (h - nu) / (lamb * gam)


def proxg(h, c, gam, lam):
    nu = nu_hat(lam, gam, h, c)
    myprint("nu = ", nu)
    nu = nu[-1]
    # print("lamb proxg", np.max(w(nu, k, c, lam, gam)))
    W = ProxEntropyNu(h, gam, lam, c, nu)
    return gam * lam * W


def nu_hat(lamb, gam, h, c):
    """
    Newton to get the value nu_hat
    """
    epsilon = 1e-9
    maxIter = 10000
    nIter = 0
    nu = [0]
    mycnt = 0
    while True:
        mycnt += 1
        # if mycnt == 100:
        #     while 1: pass
        myprint("nu = ", nu, mycnt)
        nIter += 1
        fi = phi(nu[-1], h, c, lamb, gam)
        dfi = dphi(nu[-1], h, c, lamb, gam)

        if math.isinf(dfi):
            myprint("isinf !!!!!")
            break
        # print(phi(nu[-1], k, c, lamb, gam)/dfi)
        # if dfi != 0:
        newNu = nu[-1] - fi / (dfi)
        # else: newNu = 100000
        nu.append(newNu)
        # print("nuuuuuuuu", nu[-1])
        if np.abs(newNu) > 1e50 or np.isnan(nu[-1]) or newNu < -1e50:
            newNu = (np.random.rand(1) - 0.5) * 10
            nu[-1] = newNu
            # myprint("Nan : reset nu", nu, nIter)
            # print(k)
            myprint("max k, min k", np.max(h), np.min(h))
            myprint("max c, min c", np.max(c), np.min(c))
            raise Exception('nu is nan :(')
        if nIter > maxIter or np.abs(nu[-1] - nu[-2]) < epsilon:
            break
    myprint("nu = ", nu)
    myprint("pĥi(nu)", phi(nu[-1], h, c, lamb, gam))
    return nu


def phi(nu, k, c, lamb, gam):
    myprint("lamb phi", np.max(w(nu, k, c, lamb, gam)))
    W = ProxEntropyNu(k, gam, lamb, c, nu)
    return lamb * gam * np.sum(W) - 1


def dphi(nu, k, c, lamb, gam):
    myprint(nu, lamb, gam)
    W = ProxEntropyNu(k, gam, lamb, c, nu)
    myprint("W dphi", np.max(W), "  ", np.min(W))
    myprint("retour dphi", -np.sum(W / (1 + W)))
    return -np.sum(1-1 / (1 + W))


def Lambert_W(v):
    w_matrix = np.ones(v.shape)
    u = np.inf * w_matrix

    n_iter = 0
    while np.sum(np.abs((w_matrix - u) / np.maximum(w_matrix, 1e-5)) > 1e-07) > 0 and n_iter < 100:
        u = np.copy(w_matrix)
        e = np.exp(w_matrix)
        f = w_matrix * e - v
        w_matrix = w_matrix - f / (e * (w_matrix + 1) - f * (w_matrix + 2) / (2 * w_matrix + 2))
        n_iter += 1
    return w_matrix


def myprint(*args):
    if gv.debug:
        for i in args:
            print(i, end=" ")
        print()

def ProxEntropyNu(k, gam, lam, c, nu):
    # np.where(w_n < 1e2, Lambert_W(np.exp(w_n) / (lam * gam)),
    #          w_n - np.log(lam * gam) - np.log(w_n - np.log(lam * gam)))
    limit = 200
    rho1 = 1/(lam*gam)
    rho2 = w(nu, k, c, lam, gam)
    myprint("W ", np.max(rho2), np.min(rho2))
    tau = rho2 + np.log(rho1)
    # print(tau)
    U = tau*(1 - np.log(tau)/(1+tau))
    myprint("U ", np.max(U), np.min(U))
    U[tau < limit] = Lambert_W(np.exp(tau[tau < limit]))

    myprint("U2 ", np.max(U), np.min(U))
    return U
