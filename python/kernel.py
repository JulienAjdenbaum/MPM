import numpy as np


def genD(theta, var):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta[0]), -np.sin(theta[0])],
                   [0, np.sin(theta[0]), np.cos(theta[0])]])
    Ry = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                   [0, 1, 0],
                   [-np.sin(theta[1]), 0, np.cos(theta[1])]])
    Rz = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],

                   [np.sin(theta[2]), np.cos(theta[2]), 0],
                   [0, 0, 1]])

    R = Rx @ Ry @ Rz
    print()
    return np.linalg.inv(R @ np.diag(var.T) @ R.T)


def gaussian_kernel(KERNEL_SIZE, D, mu):
    x, y, z = np.mgrid[- KERNEL_SIZE: KERNEL_SIZE + 1, -KERNEL_SIZE: KERNEL_SIZE + 1, - KERNEL_SIZE: KERNEL_SIZE + 1]
    x, y, z = x - mu[0], y - mu[1], z - mu[2]
    value = np.einsum('hjkl, hi,ijkl-> jkl', np.array([x, y, z]), D, np.array([x, y, z]))
    kernel = np.exp(-(value / 2.0))
    kernel = kernel / np.sum(kernel)
    return kernel
