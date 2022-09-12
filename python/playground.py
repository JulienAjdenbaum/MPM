#%%
from skimage import io as skio
from skimage import measure
from scipy.signal import convolve
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from matplotlib import rc
def observcoupe(values, center, title):
    plt.imshow(values[:, values.shape[1] // 2 + center, :])
    plt.title(title)
    plt.show()
    plt.imshow(values[values.shape[0] // 2 + center, :, :])
    plt.title(title)
    plt.show()


def observ(values, center, title):
    observcoupe(values, center, title)
    kernel_sizex = values.shape[0] // 2
    kernel_sizey = values.shape[1] // 2
    kernel_sizez = values.shape[2] // 2

    xi, yi, zi = np.mgrid[-kernel_sizex: kernel_sizex + values.shape[0]%2,
                          -kernel_sizey: kernel_sizey + values.shape[1]%2,
                          -kernel_sizez: kernel_sizez + values.shape[2]%2]
    print(xi.shape, yi.shape, zi.shape)
    print(values.shape)
    density = values[xi, yi, zi]
    grid = mlab.pipeline.scalar_field(xi, yi, zi, values)
    mini = density.min()
    maxi = density.max()
    mlab.pipeline.volume(grid, vmin=mini, vmax=mini + .5 * (maxi - mini))
    mlab.axes()
    mlab.title(title)
    mlab.show()
#%%
path = '/home/julin/Documents/imbilles/4um/'
imname = '1_max_810_575-630_4um_Pmax_500V_3X_10_0.497umx_0.5z.tif'
im = skio.imread(path + imname)
filter_size = 3
filter = np.ones((3, 3, 3))
imfiltered = convolve(im, filter, "same")
print(imfiltered.shape)
# H, bins = np.histogramdd(imfiltered)
plt.hist(imfiltered.flatten())
plt.yscale('log')
plt.show()
print(np.min(im), np.min(imfiltered))
print(np.max(im), np.max(imfiltered))
seuil = 0.1
imfiltered[imfiltered < seuil] = 0
imfiltered[imfiltered > seuil] = 1

#
labels, num = measure.label(imfiltered, return_num=True, connectivity=3)
# print(labels)
# print(np.max(labels))
# print(np.unique(labels))
# uniques = np.unique(labels)
regions = []
regions_size = []
# print(num)
size_min = 500
for i in range(1, num):
    size = np.sum(np.where(labels == i, 1, 0))
    print(size)
    if size > size_min:
        regions.append(i)
        regions_size.append(size)
    if i%100==0:
        print(i)
print("number of regions :", len(regions))

#%%
print(regions_size)
plt.hist(regions_size)
plt.show()
sorted_regions = np.argsort(regions_size)
print(np.max(sorted_regions))
selected_region = sorted_regions[-1]
locs = np.argwhere(labels == regions[selected_region])

xmin, ymin, zmin = np.min(locs, axis=0)
xmax, ymax, zmax = np.max(locs, axis=0)
print(xmin, ymin, zmin, xmax, ymax, zmax)
#%%
im_croped = im[xmin:xmax, ymin:ymax, zmin:zmax]
dx, dy, dz = 0.5, 0.497, 0.497
observ(im_croped, 0, "crop", (dx, dy, dz))
print("region :", i, "size :", im_croped.flatten().shape[0])
path_crops = '/home/julin/Documents/imbilles/crops/4um/'
skio.imsave(path_crops+imname, im_croped)

#%%
im = skio.imread()

