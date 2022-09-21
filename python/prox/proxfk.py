import numpy as np
from scipy.signal import convolve
import math
import prox.utils as utils
debug = False


def prox(y, k, p, D, x, mu, eps, gam, lam, alph):
    # print("p@p.T", np.max(p))
    grad = convolve(convolve(k, p, 'same') - y, p[::-1, ::-1, ::-1], "same")
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

def w(nu, k, c, lamb, gam):
    # myprint("w :", np.max(c), np.max(k), nu)
    return -1 - c + (k - nu) / (lamb * gam)


def proxg(k, c, gam, lam):
    nu = nu_hat(lam, gam, k, c)
    myprint("nu = ", nu)
    nu = nu[-1]
    # print("lamb proxg", np.max(w(nu, k, c, lam, gam)))
    w_n = w(nu, k, c, lam, gam)
    W = np.where(w_n < 1e2, Lambert_W(np.exp(w_n) / (lam * gam)),
                 w_n - np.log(lam * gam) - np.log(np.maximum(w_n - np.log(lam * gam), 1e-10)))
    return gam * lam * W


def nu_hat(lamb, gam, k, c):
    """
    Newton to get the value nu_hat
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
        newNu = nu[-1] - fi / dfi
        # else: newNu = 100000
        nu.append(newNu)
        # print("nuuuuuuuu", nu[-1])
        if np.abs(newNu) > 1e50 or np.isnan(nu[-1]) or newNu < -1e50:
            newNu = (np.random.rand(1) - 0.5) * 10
            nu[-1] = newNu
            # myprint("Nan : reset nu", nu, nIter)
            # print(k)
            myprint("max k, min k", np.max(k), np.min(k))
            myprint("max c, min c", np.max(c), np.min(c))
            raise Exception('nu is nan :(')
        if nIter > maxIter or np.abs(nu[-1] - nu[-2]) < epsilon:
            break
    myprint(nu)
    return nu


def phi(nu, k, c, lamb, gam):
    myprint("lamb phi", np.max(w(nu, k, c, lamb, gam)))
    w_n = w(nu, k, c, lamb, gam)
    W = np.where(w_n < 1e2, Lambert_W(np.exp(w_n) / (lamb * gam)),
                 w_n - np.log(lamb * gam) - np.log(np.maximum(w_n - np.log(lamb * gam), 1e-10)))
    return lamb * gam * np.sum(W) - 1


def dphi(nu, k, c, lamb, gam):
    w_n = w(nu, k, c, lamb, gam)
    W = np.where(w_n < 1e2, Lambert_W(np.exp(w_n)/(lamb*gam)), w_n - np.log(lamb* gam) - np.log(np.maximum(w_n-np.log(lamb* gam), 1e-10)))
    myprint("W dphi", np.max(W), "  ", np.min(W))
    myprint("retour dphi", -np.sum(W / (1 + W)))
    return -np.sum(W / (1 + W)) / (lamb * gam)


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
    if debug:
        for i in args:
            print(i, end=" ")
        print()
