import numpy as np


def prox(x, k, D, mu, gam, lam, eps):
    i_q = np.eye(D.shape[0])
    return np.linalg.inv(i_q + gam * lam * np.sum(k) * (D + eps * i_q)) @ \
           (mu + gam * lam * np.einsum("ijk, ijkl -> l", k, np.einsum('ij, abcj -> abci', (D + eps * i_q), x)))
