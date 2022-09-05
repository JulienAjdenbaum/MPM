import numpy as np
import observation3D

def make_sphere(sphere_size, volume_size, interpolated):
    x, y, z = np.mgrid[- volume_size : volume_size + 1, -volume_size : volume_size + 1, - volume_size : volume_size + 1]
    if interpolated:
        sphere = np.zeros((x.shape[0], y.shape[0], z.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                for k in range(x.shape[0]):
                    if np.sqrt(x[i, j, k] ** 2 + y[i, j, k] ** 2 + z[i, j, k] ** 2) < sphere_size + 1:
                        if np.sqrt(x[i, j, k] ** 2 + y[i, j, k] ** 2 + z[i, j, k] ** 2) < sphere_size:
                            sphere[i, j, k] = 1
                        else:
                            sphere[i, j, k] = (1 + sphere_size - np.sqrt(x[i, j, k] ** 2 + y[i, j, k] ** 2 + z[i, j, k] ** 2))**2
        return sphere
    return np.where(np.sqrt(x**2 + y**2 + z**2)<sphere_size, 1, 0)


# values = make_sphere(8, 10)
# observation3D.observ(values)