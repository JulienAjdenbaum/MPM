import numpy as np
import random
from gen_observation import gen_observation
import global_variables as gv
import kernel as kernel
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

Y_big = np.zeros(taille_image)
Y_small = np.zeros(taille_image)
for i in range(n_billes):
    centre = np.array(np.unravel_index(random.randint(0, int(np.prod(taille_image))), taille_image))
    window_size = max(int(4 * np.max(variance)), int(np.max(4 * gv.sphere_size * np.ones(3) / gv.resolution)))
    gv.kernel_size = 2 * window_size * np.ones(3)
    if np.all(centre - window_size >= 0) and np.all(centre + window_size < taille_image):
        Y_big[centre[0] - window_size: centre[0] + window_size,
        centre[1] - window_size: centre[1] + window_size,
        centre[2] - window_size: centre[2] + window_size] = gen_observation([0, 0, 0], C, 0, sphere_size=1)[0]
        Y_small[centre[0] - window_size: centre[0] + window_size,
        centre[1] - window_size: centre[1] + window_size,
        centre[2] - window_size: centre[2] + window_size] = gen_observation([0, 0, 0], C, 0, sphere_size=0.2)[0]

Y_big = (Y_big + np.random.randn(taille_image[0], taille_image[1], taille_image[2])*sigma)*4096
Y_small = (Y_small + np.random.randn(taille_image[0], taille_image[1], taille_image[2])*sigma)*4096

gv.plot = True
plt.close("all")
observation3D.observ(Y_big, 0, "Y")
skio.imsave("images/Y_big.tif", Y_big)
skio.imsave("images/Y_small.tif", Y_small)
file = open("images/C", "w")
file.write(str(C))
file.close()
