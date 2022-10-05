import numpy as np
import global_variables as gv

def make_sphere(sphere_size, volume_size, a=1):
    rayon = gv.sphere_size/2
    print("rayon =", rayon)
    x, y, z = np.mgrid[- volume_size[0]: volume_size[0] + 1,
                       - volume_size[1]: volume_size[1] + 1,
                       - volume_size[2]: volume_size[2] + 1]
    x = x*gv.resolution[0]
    y = y*gv.resolution[1]
    z = z*gv.resolution[2]
    if gv.interpolated:
        # raise Exception('interpolated is True')
        sphere = np.zeros((x.shape[0], x.shape[1], z.shape[2]))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    if np.sqrt((np.abs(x[i, j, k])-gv.resolution[0]) ** 2
                               + (np.abs(y[i, j, k])-gv.resolution[1]) ** 2
                               + (np.abs(z[i, j, k])-gv.resolution[2]) ** 2) <= rayon \
                        or np.sqrt((x[i, j, k]) ** 2
                                   + (y[i, j, k]) ** 2
                                   + (z[i, j, k]) ** 2)< rayon:
                        if np.sqrt(x[i, j, k] ** 2 + y[i, j, k] ** 2 + z[i, j, k] ** 2) <= rayon:
                            sphere[i, j, k] = a
                        else:
                            sphere[i, j, k] = a*(1-np.sqrt((np.abs(x[i, j, k])-gv.resolution[0]) ** 2
                               + (np.abs(y[i, j, k])-gv.resolution[1]) ** 2
                               + (np.abs(z[i, j, k])-gv.resolution[2]) ** 2)/rayon)
        return sphere
    sphere = np.where(np.sqrt(x ** 2 + y** 2 + z ** 2) <= rayon, a, 0)
    return sphere
# values = make_sphere(8, 10)
# observation3D.observ(values)
