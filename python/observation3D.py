import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from matplotlib import rc

rc('text', usetex=True)


# def observ(values, crop):
#     k = values.shape[0]//2
#     X, Y, Z = np.mgrid[- k : k + 1, -k : k + 1, - k : k + 1]
#     fig = go.Figure(data=go.Volume(
#         x=X.flatten(),
#         y=Y.flatten(),
#         z=Z.flatten(),
#         value=values.flatten(),
#         opacity=0.1, # needs to be small to see through all surfaces
#         surface_count=100, # needs to be a large number for good volume rendering
#         ))
#     fig.show()

# def observ(values, seuil):
#
#     k = values.shape[0] // 2
#     X, Y, Z = np.mgrid[- k : k + 1, -k : k + 1, - k : k + 1]
#     valx, valy, valz = (values>seuil).nonzero()
#     # print(np.max(val))
#     ax = plt.axes(projection='3d')
#     print(values[valx, valy, valz])
#     ax.scatter3D(valx, valy, valz, c=values[valx, valy, valz], cmap='Blues', s=3)
#     ax.set_xlim(0, 21)
#     ax.set_ylim(0, 21)
#     ax.set_zlim(0, 21)
#     plt.show()
#     print("ploted")

def observcoupe(values, center, title):
    plt.imshow(values[:, values.shape[1] // 2 + center, :])
    plt.title(title + " - vue latérale ")
    plt.show()
    plt.imshow(values[values.shape[0] // 2 + center, :, :])
    plt.title(title + " - vue du dessus")
    plt.show()


def observ(values, center, title):
    observcoupe(values, center, title)
    kernel_sizex = values.shape[0] // 2
    kernel_sizey = values.shape[1] // 2
    kernel_sizez = values.shape[2] // 2
    xi, yi, zi = np.mgrid[-kernel_sizex: kernel_sizex + values.shape[0] % 2,
                 -kernel_sizey: kernel_sizey + values.shape[1] % 2,
                 -kernel_sizez: kernel_sizez + values.shape[2] % 2]
    density = values[xi, yi, zi]
    grid = mlab.pipeline.scalar_field(xi, yi, zi, values)
    mini = density.min()
    maxi = density.max()
    mlab.pipeline.volume(grid, vmin=mini, vmax=mini + .5 * (maxi - mini))
    mlab.axes()
    mlab.title(title)
    mlab.show()
