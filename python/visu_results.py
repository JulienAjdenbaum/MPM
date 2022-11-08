import numpy as np
import global_variables as gv
import os
import matplotlib.pyplot as plt
import time
from skimage import io as skio
from observation3D import observ
from kernel import gaussian_kernel
from scipy.signal import fftconvolve

experiment = 'last'

if experiment == 'last':
    if os.listdir(gv.save_path) == []:
        n = 0
    else:
        print(np.max(list(map(int, os.listdir(gv.save_path)))))
        n = np.max(list(map(int, os.listdir(gv.save_path))))
    experiment = n

gv.save_path = gv.save_path + str(experiment) + "/"
file = open(gv.save_path+"settings.txt", "r")
print(gv.save_path+"settings.txt")
file_content = file.read()
file.close()
print(file_content)

val_path = gv.save_path + "values/"

im = np.array(np.load(val_path + 'im.npy'))
gv.kernel_size = np.array(im.shape)
# observ(im, 0, 'Image')

D = np.array(np.load(val_path + 'D.npy'))
try:
    mu = np.array(np.load(val_path + 'mu.npy'))
except:
    print("mu not found, setting it to 0")
    mu = np.array([0, 0, 0])
k_args = gaussian_kernel(D, mu)
observ(k_args, 0, 'k_args')

p = np.array(np.load(val_path + 'p.npy'))
observ(p, 0, 'Bille vraie')

kcomvolvep = fftconvolve(k_args, p, "same")
print("map p : ", np.max(p))

observ(p, 0, 'p')
print(type(p))
print(p.shape)
print((type(im)))
print(im.shape)
plt.imshow(p[26, :,:])
plt.show()
plt.imshow(im[26, :,:])
plt.show()
skio.imsave(val_path+ "bille_simulee.tif", 1000*p)
skio.imsave(val_path+ "crop_bille.tif", im)
skio.imsave(val_path+ "PSF_args.tif", k_args)
skio.imsave(val_path+ "kconvolvep.tif", kcomvolvep)


plt_path = gv.save_path + "plots/"

for plot in os.listdir(plt_path):
    im = skio.imread(plt_path+plot)
    plt.imshow(im)
    plt.show()




# print(D)

# print(sigma)
# print("D  :", D)
# print("Covariance : ", np.linalg.eig(sigma)[0])
# print("sigma : ", np.sqrt(np.linalg.eig(sigma)[0]))
sigma = np.linalg.inv(D)
print("FWMH : ", np.sqrt(np.linalg.eig(sigma)[0])*gv.resolution*(2*np.sqrt(2*np.log(2))))# print(D)
# sigma = np.linalg.inv(D)
# print(sigma)*)