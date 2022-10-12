
from skimage import io as skio
from skimage import measure
from scipy.signal import fftconvolve
import numpy as np
from make_sphere import make_sphere
from observation3D import observ, observ_distri
from main import from_bille, kernel, gen_observation
from prox.utils import get_barycentre, get_a
import time
import global_variables as gv
import scipy

#%%
# def pad(im):
#     xg, yg, zg = get_barycentre(im)
#     maxi = int(np.ceil(np.max((xg, yg, zg, im.shape[0]-xg, im.shape[1]-yg, im.shape[2]-zg))))
#     # print(maxi)
#     im_padded = np.zeros((2*maxi+1, 2*maxi+1, 2*maxi+1))
#     # print(im_padded.shape)
#     # print(int(im_padded.shape[0] // 2 - xg),
#     #       int(im_padded.shape[1] // 2 - yg),
#     #       int(im_padded.shape[2] // 2 - zg))
#     # print(int(im_padded.shape[0] // 2 - xg)+im.shape[0],
#     #       int(im_padded.shape[1] // 2 - yg)+im.shape[1],
#     #       int(im_padded.shape[2] // 2 - zg)+im.shape[2])
#
#     im_padded[int(im_padded.shape[0]//2-xg)+1:int(im_padded.shape[0]//2-xg)+im.shape[0]+1,
#               int(im_padded.shape[1]//2-yg)+1:int(im_padded.shape[1]//2-yg)+im.shape[1]+1,
#               int(im_padded.shape[2]//2-zg)+1:int(im_padded.shape[2]//2-zg)+im.shape[2]+1] = im
#     # print(get_barycentre(im_padded))
#     # observ(im_padded, 0, "padded")
#     return im_padded

def pad(im):
    maxi = np.max(im.shape)
    im_new = np.zeros((maxi, maxi, maxi))
    debut = (np.array((maxi, maxi, maxi))-np.array(im.shape))//2
    im_new[debut[0]:debut[0]+im.shape[0],
           debut[1]:debut[1]+im.shape[1],
           debut[2]:debut[2]+im.shape[2]] = im
    print(im_new.shape)
    return im_new


def pipeline(im):
    # im = im/np.max(im)
    # a = get_a(im)
    # print('a = ', a)
    # if reel:
    #     im = im/a
    #     print(np.max(im))
    # im = pad(im)
    if gv.plot:
        observ(im, 0, "image")
    p = make_sphere()
    if gv.plot:
        observ(p, 0, 'p')
        # observ(im, 0, "im")
    return from_bille(p, im)


t = time.time()

if gv.reel:
    crop_file = '/home/julin/Documents/imbilles/crops/4um/1_max_810_575-630_4um_Pmax_500V_3X_10_0.497umx_0.5z/2.tif'
    crop_file = '/home/julin/Documents/imbilles/crops/1um2/860_495-540_0.049xy_0.5z_2/0.tif'

    Y = skio.imread(crop_file)
    observ_distri(Y, gv.resolution, 'Y distribution')
    gv.kernel_size = np.array(Y.shape)

if gv.simulation:
    C = np.divide(gv.FWMH/(2*np.sqrt(2*np.log(2))), gv.resolution)
    # C = np.array([1, 1, 5])

    mutrue = np.array([0, 0, 0])
    Dtrue = kernel.genD(gv.angle, C)
    print(Dtrue)
    Y, ktrue, p = gen_observation(mutrue, Dtrue, sigma_noise=0.1)
    observ_distri(ktrue, gv.resolution, 'k_true distribution')
    # print(Y)
lamb = 1
chosen_D, chosen_mu, k_args, chosen_k, p = pipeline(Y)
if gv.simulation:
    observ(ktrue, 0, "ktrue")


observ(k_args, 0, "kargs")
observ(chosen_k, 0, "k_est")
observ(fftconvolve(k_args, p, "same"), 0, "k convolué avec p")
observ_distri(k_args, gv.resolution, 'k_args distribution')
