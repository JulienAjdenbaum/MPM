import numpy as np
import global_variables as gv

def make_sphere(sphere_size, volume_size, a=1):
    rayon = sphere_size//2
    x, y, z = np.mgrid[- volume_size: volume_size + 1, -volume_size: volume_size + 1, - volume_size: volume_size + 1]
    if gv.interpolated:
        raise Exception('interpolated is True')
        sphere = np.zeros((x.shape[0], y.shape[0], z.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                for k in range(x.shape[0]):
                    if np.sqrt(x[i, j, k] ** 2 + y[i, j, k] ** 2 + z[i, j, k] ** 2) < rayon + 1:
                        if np.sqrt(x[i, j, k] ** 2 + y[i, j, k] ** 2 + z[i, j, k] ** 2) < rayon:
                            sphere[i, j, k] = a
                        else:
                            sphere[i, j, k] = a*(1 + rayon
                                               - np.sqrt(x[i, j, k] ** 2 + y[i, j, k] ** 2 + z[i, j, k] ** 2)) ** 2
        return sphere
    return np.where(np.sqrt(x ** 2 + y ** 2 + z ** 2) < rayon, a, 0)

# values = make_sphere(8, 10)
# observation3D.observ(values)
