import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from matplotlib import rc
from prox.utils import get_barycentre

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


def observ_distri(values, resolutions, titre):
    centre = np.array(get_barycentre(values)).astype(int)
    idx = (np.arange(len(values[0, :, 0])) - centre[1]) * resolutions[1]
    idy = (np.arange(len(values[0, 0, :])) - centre[2]) * resolutions[2]
    idz = (np.arange(len(values[:, 0, 0])) - centre[0]) * resolutions[0]
    plt.plot(idy, values[centre[0], centre[1], :])
    mi_hauteur = 0.5 * np.max(values[centre[0], centre[1], :])
    mi_hauteur_min = np.min(np.argwhere(values[centre[0], centre[1], :] >= mi_hauteur)) - 0.5
    mi_hauteur_max = np.max(np.argwhere(values[centre[0], centre[1], :] >= mi_hauteur)) + 0.5
    mi_hauteur_min = (mi_hauteur_min - centre[2]) * resolutions[2]
    mi_hauteur_max = (mi_hauteur_max - centre[2]) * resolutions[2]
    plt.plot([mi_hauteur_min, mi_hauteur_max], [mi_hauteur, mi_hauteur], marker="o")
    FWMH = (mi_hauteur_max - mi_hauteur_min)
    m = (mi_hauteur_max + mi_hauteur_min) / 2
    plt.text(m, mi_hauteur * 1.05, "FWMH = " + str(FWMH) + "$\mu m$", ha="center", fontsize="small")
    plt.title(titre + " - distribution selon y")
    plt.xlabel("axe y ($\mu m$)")
    plt.show()
    plt.plot(idx, values[centre[0], :, centre[2]])
    mi_hauteur = 0.5 * np.max(values[centre[0], :, centre[2]])
    mi_hauteur_min = np.min(np.argwhere(values[centre[0], :, centre[2]] >= mi_hauteur)) - 0.5
    mi_hauteur_max = np.max(np.argwhere(values[centre[0], :, centre[2]] >= mi_hauteur)) + 0.5
    mi_hauteur_min = (mi_hauteur_min - centre[1]) * resolutions[1]
    mi_hauteur_max = (mi_hauteur_max - centre[1]) * resolutions[1]
    plt.plot([mi_hauteur_min, mi_hauteur_max], [mi_hauteur, mi_hauteur], marker="o")
    plt.title(titre + " - distribution selon x")
    FWMH = (mi_hauteur_max - mi_hauteur_min)
    m = (mi_hauteur_max + mi_hauteur_min) / 2
    plt.text(m, mi_hauteur * 1.05, "FWMH = " + str(FWMH) + "$\mu m$", ha="center", fontsize="small")
    plt.xlabel("axe x ($\mu m$)")
    plt.show()
    plt.plot(idz, values[:, centre[1], centre[2]])
    mi_hauteur = 0.5 * np.max(values[:, centre[1], centre[2]])
    print(mi_hauteur)
    mi_hauteur_min = np.min(np.argwhere(values[:, centre[1], centre[2]] >= mi_hauteur)) - 0.5
    mi_hauteur_max = np.max(np.argwhere(values[:, centre[1], centre[2]] >= mi_hauteur)) + 0.5
    mi_hauteur_min = (mi_hauteur_min - centre[0]) * resolutions[0]
    mi_hauteur_max = (mi_hauteur_max - centre[0]) * resolutions[0]
    print(mi_hauteur_min)
    print(mi_hauteur_max)
    plt.plot([mi_hauteur_min, mi_hauteur_max], [mi_hauteur, mi_hauteur], marker="o")
    plt.title(titre + " - distribution selon z")
    FWMH = (mi_hauteur_max - mi_hauteur_min)
    m = (mi_hauteur_max + mi_hauteur_min) / 2
    plt.text(m, mi_hauteur * 1.05, "FWMH = " + str(FWMH) + "$\mu m$", ha="center", fontsize="small")
    plt.xlabel("axe z ($\mu m$)")
    plt.show()
