import numpy as np


def prox(D, k, x, mu, eps, lam, gam):
    xmu = x - mu
    xmu = xmu.reshape(xmu.shape[0] ** 3, 3, 1)
    kflat = k.copy().flatten()
    m = 1 / 2 * lam * gam * np.sum(kflat)
    S = 1 / 2 * lam * gam * np.einsum('a, aij, akj -> ik', kflat, xmu, xmu)
    # print("S = ", np.round(S, 2))
    # print("D = ", np.round(D, 2))
    # print("D-S", np.round(D-S, 2))
    # print(m)

    w, V = np.linalg.eigh(D - S)
    # print("VwV.T",V @ np.diag(w) @ V.T)
    # print("2*w", 2*np.round(np.diag(w), 2))
    # print("sqrt", 2 * np.diag(np.sqrt((w+eps)**2+4*m)))
    # print("max", np.diag(np.maximum(w-eps+np.sqrt((w+eps)**2+4*m), 0))/2)

    return np.abs(1 / 2 * V @ np.diag(np.maximum(w - eps + np.sqrt((w + eps) ** 2 + 4 * m), 0)) @ V.T)

    # méthode implémentée dans la pipeline matlab mais incomprise :/
    #
    # sumk = np.sum(k)
    # xmu = x-mu
    # xmu = xmu.reshape(xmu.shape[0]**3, 3, 1)
    # kflat = k.copy().flatten()
    # S = np.einsum('a, aij, akj -> ik', kflat, xmu, xmu)
    # M = D - 1/2*gam*lam*S
    # m0 = 1/2*gam*lam*sumk
    # ret = 1/2*(M+np.sqrt(M**2+4*m0*np.eye(3)))
    # print("ret", ret)
    # return ret
