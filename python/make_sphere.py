import numpy as np
import global_variables as gv
import prox.utils as utils
from scipy.signal import convolve
def make_sphere():
    rayon = gv.sphere_size/2
    print("rayon =", rayon)
    x, y, z, X = utils.mymgrid()
    if gv.interpolated:
        # raise Exception('interpolated is True')
        sphere = np.zeros((x.shape[0], x.shape[1], z.shape[2]))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    b = np.sqrt((np.abs(x[i, j, k] * gv.resolution[0]) - gv.resolution[0]) ** 2
                                + (np.abs(y[i, j, k] * gv.resolution[1]) - gv.resolution[1]) ** 2
                                + (np.abs(z[i, j, k] * gv.resolution[2]) - gv.resolution[2]) ** 2)
                    if np.sqrt((x[i, j, k] * gv.resolution[0]) ** 2
                               + (y[i, j, k] * gv.resolution[1]) ** 2
                               + (z[i, j, k] * gv.resolution[2]) ** 2) <= rayon:
                        sphere[i, j, k] = 1
                    elif b <= rayon:
                        a = np.sqrt((np.abs(x[i, j, k] * gv.resolution[0])) ** 2
                                    + (np.abs(y[i, j, k] * gv.resolution[1])) ** 2
                                    + (np.abs(z[i, j, k] * gv.resolution[2])) ** 2)
                        sphere[i, j, k] = (rayon-b)/(a-b)
        return sphere
    print(np.max((x/gv.resolution[0])))
    sphere = np.where(np.sqrt(((x-1)*gv.resolution[0]) ** 2 + ((y-1)*gv.resolution[1]) ** 2 + ((z-1)*gv.resolution[2]) ** 2) <= rayon, 1, 0)
    return sphere
# values = make_sphere(8, 10)
# observation3D.observ(values)
