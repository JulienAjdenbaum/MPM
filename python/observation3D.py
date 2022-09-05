import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mayavi import mlab

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
    plt.imshow(values[values.shape[0]//2+center,:,:])
    plt.title(title)
    plt.show()


def observ(values, center, title):
    observcoupe(values, center, title)
    # kernel_size = values.shape[0]//2
    # xi, yi, zi = np.mgrid[- kernel_size: kernel_size + 1,
    #           -kernel_size: kernel_size + 1,
    #           - kernel_size: kernel_size + 1]
    # density = values[xi, yi, zi]
    # grid = mlab.pipeline.scalar_field(xi, yi, zi, values)
    # min = density.min()
    # max = density.max()
    # mlab.pipeline.volume(grid, vmin=min, vmax=min + .5 * (max - min))
    # mlab.axes()
    # mlab.title(title)
    # mlab.show()