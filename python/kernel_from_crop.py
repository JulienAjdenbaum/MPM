from skimage import io as skio
from skimage import measure
from scipy.signal import convolve
import numpy as np
from make_sphere import make_sphere
from observation3D import observ
from main import from_bille

def get_barycentre(im):
    xi, yi, zi = np.mgrid[0:im.shape[0],
                 0:im.shape[1],
                 0:im.shape[2]]
    xg = np.sum(np.multiply(xi, im)) / np.sum(im)
    yg = np.sum(np.multiply(yi, im)) / np.sum(im)
    zg = np.sum(np.multiply(zi, im)) / np.sum(im)
    return (xg, yg, zg)

def pad(im):
    xg, yg, zg = get_barycentre(im)
    maxi = int(np.ceil(np.max((xg, yg, zg, im.shape[0]-xg, im.shape[1]-yg, im.shape[2]-zg))))
    # print(maxi)
    im_padded = np.zeros((2*maxi+1, 2*maxi+1, 2*maxi+1))
    # print(im_padded.shape)
    # print(int(im_padded.shape[0] // 2 - xg),
    #       int(im_padded.shape[1] // 2 - yg),
    #       int(im_padded.shape[2] // 2 - zg))
    # print(int(im_padded.shape[0] // 2 - xg)+im.shape[0],
    #       int(im_padded.shape[1] // 2 - yg)+im.shape[1],
    #       int(im_padded.shape[2] // 2 - zg)+im.shape[2])

    im_padded[int(im_padded.shape[0]//2-xg)+1:int(im_padded.shape[0]//2-xg)+im.shape[0]+1,
              int(im_padded.shape[1]//2-yg)+1:int(im_padded.shape[1]//2-yg)+im.shape[1]+1,
              int(im_padded.shape[2]//2-zg)+1:int(im_padded.shape[2]//2-zg)+im.shape[2]+1] = im
    # print(get_barycentre(im_padded))
    # observ(im_padded, 0, "padded")
    return im_padded

crop_file = '/home/julin/Documents/imbilles/crops/4um/1_max_810_575-630_4um_Pmax_500V_3X_10_0.497umx_0.5z/2.tif'
im = skio.imread(crop_file)
im = im/np.max(im)
print(np.max(im))
# observ(im, 0, "crop")
# print(im.shape)
# print(get_barycentre(im))
im = pad(im)
real_sphere_size = 4/0.5
voxel_size = (0.5, 0.497, 0.497)
p = make_sphere(real_sphere_size//2, im.shape[0]//2, False)
observ(p, 0, "Bille simulée")
observ(im, 0, "Observation bille")
chosen_D, chosen_mu, k_args, chosen_k, norm1, norm2 = from_bille(1, True, p, im)
