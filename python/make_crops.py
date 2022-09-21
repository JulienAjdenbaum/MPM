from skimage import io as skio
from skimage import measure
from scipy.signal import convolve
import numpy as np
import os
import shutil

path_ims = '/home/julin/Documents/imbilles/1um/'
path_crops = '/home/julin/Documents/imbilles/crops/1um/'
dirs = os.listdir(path_ims)

for imname in dirs:
    print("opening image : ", imname)
    im = skio.imread(path_ims + imname)
    print(im.shape)
    while(1): pass
    filter_size = 3
    filter = np.ones((3, 3, 3))
    imfiltered = convolve(im, filter, "same")

    seuil = 0.1
    imfiltered[imfiltered < seuil] = 0
    imfiltered[imfiltered > seuil] = 1

    labels, n_regions = measure.label(imfiltered, return_num=True)

    regions = []
    regions_size = []
    print("Nombre de pré-régions : ", n_regions)

    size_min = 300
    for i in range(1, n_regions):
        size = np.sum(np.where(labels == i, 1, 0))
        if size > size_min:
            regions.append(i)
            regions_size.append(size)
    print("Nombre de régions sélectionnées :", len(regions))

    try:
        os.makedirs(path_crops + imname[:-4])
    except FileExistsError:
        shutil.rmtree(path_crops + imname[:-4], ignore_errors=True)
        os.makedirs(path_crops + imname[:-4])

    for i in range(len(regions)):
        selected_region = regions[i]
        locs = np.argwhere(labels == selected_region)
        xmin, ymin, zmin = np.min(locs, axis=0)
        xmax, ymax, zmax = np.max(locs, axis=0)
        im_croped = im[xmin:xmax, ymin:ymax, zmin:zmax]
        print("region :", i, "size :", im_croped.flatten().shape[0])

        print(os.listdir())
        skio.imsave(path_crops + imname[:-4] + "/" + str(i) + ".tif", im_croped)

