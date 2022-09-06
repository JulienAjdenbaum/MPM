import numpy as np
from scipy.signal import convolve
import math

debug = False


def prox(y, k, p, c, gam, lam, alph):
    # print("p@p.T", np.max(p))
    grad = convolve(convolve(k, p, 'same') - y, p, "same")
    # grad = convolve(k, p@p.T, "same")-y@p
    # print("grad = ", np.linalg.norm(grad))
    forward = k - alph * grad
    # print("forward = ", np.linalg.norm(forward-k))
    # print("diff", np.linalg.norm(k-forward))
    return proxg(forward, c, gam, lam)


def mylambertw(w_n):
    myprint("deb lambert", np.max(w_n), np.min(w_n))
    myprint("deb lambert", np.max(np.exp(w_n)), np.min(np.exp(w_n)))
    v = np.where(w_n < 1e2, np.real(Lambert_W(np.exp(w_n))), w_n - np.log(np.maximum(w_n, 1e-10)))
    myprint("fin lambert", np.max(v), np.min(v))
    return v


def proxg(k, c, gam, lam):
    nu = nu_hat(lam, gam, k, c)
    myprint("nu = ", nu)
    nu = nu[-1]
    # print("lamb proxg", np.max(w(nu, k, c, lam, gam)))
    return np.real(gam * lam * mylambertw(w(nu, k, c, lam, gam) / (lam * gam))), nu


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


def w(nu, k, c, lamb, gam):
    # myprint("w :", np.max(c), np.max(k), nu)
    return -1 - c + (k - nu) / (lamb * gam)


def phi(nu, k, c, lamb, gam):
    myprint("lamb phi", np.max(w(nu, k, c, lamb, gam)))
    return lamb * gam * np.sum(mylambertw(w(nu, k, c, lamb, gam) / (lamb * gam))) - 1


def dphi(nu, k, c, lamb, gam):
    W = mylambertw(w(nu, k, c, lamb, gam) / (lamb * gam))
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
