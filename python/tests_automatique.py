import numpy as np
import random
from MPM.python.gen_observation import gen_observation
import MPM.python.global_variables as gv
import MPM.python.kernel as kernel
import observation3D
import matplotlib.pyplot as plt
from skimage import io as skio


taille_image = np.array([512, 512, 128])
n_billes = 10
sigma = 0.1

FWHM = np.random.uniform(0.1, 4, 3)
variance = np.divide(FWHM / (2 * np.sqrt(2 * np.log(2))), gv.resolution)
angles = np.random.uniform(0, 2 * np.pi, 3)

C = kernel.genC(angles, variance)
mu = np.random.random(3) * 2 - 1

Y = np.zeros(taille_image)
for i in range(n_billes):
    centre = np.array(np.unravel_index(random.randint(0, int(np.prod(taille_image))), taille_image))
    window_size = max(int(4 * np.max(variance)), int(np.max(4 * gv.sphere_size * np.ones(3) / gv.resolution)))
    gv.kernel_size = 2 * window_size * np.ones(3)
    if np.all(centre - window_size >= 0) and np.all(centre + window_size < taille_image):
        Y[centre[0] - window_size: centre[0] + window_size,
        centre[1] - window_size: centre[1] + window_size,
        centre[2] - window_size: centre[2] + window_size] = gen_observation([0, 0, 0], C, 0)[0]

Y = (Y + np.random.randn(taille_image[0], taille_image[1], taille_image[2])*sigma)*4096
gv.plot = True
plt.close("all")
print(Y.shape)
print(Y)
observation3D.observ(Y, 0, "Y")
skio.imsave("images/Y.tif", Y)
file = open("C", "w")
file.write(str(C))
file.close()
