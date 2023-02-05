import numpy as np
import random
from gen_observation import gen_observation
import global_variables as gv
import kernel as kernel
import observation3D
import matplotlib.pyplot as plt
from skimage import io as skio

np.random.seed(0)
taille_image = np.array([512, 512, 128])
n_billes = 10
sigma = 0.01

# FWHM = np.random.uniform(0, 3, 3)
FWHM = np.array([0.3, 0.3, 1.5])
print("FWHM", FWHM)
variance = (np.divide(FWHM / (2*np.sqrt(2 * np.log(2))), gv.resolution))**2
print("variance", variance)
angles = np.random.uniform(0, 2 * np.pi, 3)
angles = [0, 0, 0]
C = kernel.genC(angles, variance)
print("C", C)
mu = np.random.random(3) * 2 - 1

Y_big = np.zeros(taille_image)
Y_small = np.zeros(taille_image)
for i in range(n_billes):
    centre = np.array(np.unravel_index(random.randint(0, int(np.prod(taille_image))), taille_image))
    # print(4*np.max(variance))
    # print(np.max(4 * gv.sphere_size * np.ones(3) / gv.resolution))
    # print("maaax")
    window_size = max(int(2 * np.max(np.sqrt(variance))), int(np.max(2 * gv.sphere_size * np.ones(3) / gv.resolution)))
    for i in range(3):
        if centre[i]<window_size:
            centre[i]=window_size
        elif centre[i]+window_size>taille_image[i]:
            centre[i]=taille_image[i]-window_size
    print("window size", window_size)
    gv.kernel_size = 2 * window_size * np.ones(3)
    # print('aaa', gv.kernel_size)
    # print(2 * window_size * np.ones(3))
    # print(centre)
    # print(window_size)
    # print(taille_image)
    a, b, c = np.shape(Y_big[centre[0] - window_size: centre[0] + window_size,
                            centre[1] - window_size: centre[1] + window_size,
                            centre[2] - window_size: centre[2] + window_size])
    Y_big[centre[0] - window_size: centre[0] + window_size,
    centre[1] - window_size: centre[1] + window_size,
    centre[2] - window_size: centre[2] + window_size] = gen_observation([0, 0, 0], C, 0, sphere_size=1)[0][:a, :b, :c]
    print("centre", centre)
    Y_small[centre[0] - window_size: centre[0] + window_size,
    centre[1] - window_size: centre[1] + window_size,
    centre[2] - window_size: centre[2] + window_size] = gen_observation([0, 0, 0], C, 0, sphere_size=0.2)[0][:a, :b, :c]
    plt.close('all')
    # plt.imshow(Y_small[centre[0] - window_size: centre[0] + window_size,
    # centre[1] - window_size: centre[1] + window_size,
    # centre[2] - window_size: centre[2] + window_size][:, 50, :])
    # plt.show()

Y_big = (Y_big + np.random.randn(taille_image[0], taille_image[1], taille_image[2])*sigma)*4096
Y_small = (Y_small + np.random.randn(taille_image[0], taille_image[1], taille_image[2])*sigma)*4096

gv.plot = True
print("Y_small min = ", np.min(Y_small))
print("Y_big min = ", np.min(Y_big))
print(Y_small[0])
print(Y_big[0])
plt.close("all")
# observation3D.observ(Y_big, 0, "Y")
skio.imsave("images/Y_big.tif", Y_big)
skio.imsave("images/Y_small.tif", Y_small)
file = open("C", "w")
file.write(str(C))
file.close()
