import numpy as np
from scipy.signal import fftconvolve
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter
from cupyx.scipy.signal import fftconvolve as cupyxfftconvolve
import time
import cupy as cp

data_size = 330
# use scipy filtering functions designed to apply kernels to isolate a 1d gaussian kernel
kernel_base = np.ones(shape=(5))
kernel_1d = gaussian_filter(kernel_base, sigma=1, mode='constant')
kernel_1d = kernel_1d / np.sum(kernel_1d)

# make the 3d kernel that does gaussian convolution in z axis only
kernel_3d = cp.zeros(shape=(1, 1, 5,))
kernel_3d[0, 0, :] = cp.array(kernel_1d)

# generate random data
data = cp.random.random(size=(data_size, data_size, data_size))

# define a function for loop based convolution for easy timeit invocation
# def convolve_with_loops(data):
#     nx, ny, nz = data.shape
#     convolved=np.zeros((nx, ny, nz))
#     for i in range(0, nx):
#         for j in range(0, ny):
#             convolved[i,j,:]= fftconvolve(data[i, j, :], kernel_1d, mode='same')
#     return convolved

# compute the convolution two diff. ways: with loops (first) or as a 3d convolution (2nd)
# convolved = convolve_with_loops(data)
# convolved_2 = fftconvolve(data, kernel_3d, mode='same')

# raise an error unless the two computations return equivalent results
# assert np.all(np.isclose(convolved, convolved_2))

# time the two routes of the computation
# t = time.time()
# for i in range(10):
#     print(i)
#     fftconvolve(data, kernel_3d, mode='same')
# print(time.time()-t)
t = time.time()
for i in range(10):
    print(i)
    convolve(data, kernel_3d, mode='same')
print(time.time()-t)

data = cp.array(data)
kernel_3d = cp.array(kernel_3d)

t = time.time()
for i in range(10):
    print(i)
    cupyxfftconvolve(data, kernel_3d, mode='same')
print(time.time()-t)