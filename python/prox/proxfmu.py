import numpy as np
import global_variables as gv

def prox(x, h, D, mu, lam, eps):
    gam = gv.gam_mu
    i_q = np.eye(D.shape[0])
    return np.linalg.inv(i_q + gam * lam * np.sum(h) * (D + eps * i_q)) @ \
           (mu + gam * lam * np.einsum("ijk, ijkl -> l", h, np.einsum('ij, abcj -> abci', (D + eps * i_q), x)))
