import matplotlib.pyplot as plt
from skimage import io as skio
from skimage import measure
from scipy.signal import fftconvolve, wiener
import numpy as np
from make_sphere import make_sphere
from observation3D import observ, observ_distri
from PSFEstimation import PSFEstimation, kernel
from MPM.python.gen_observation import gen_observation
from prox.utils import get_barycentre, get_a
import time
import global_variables as gv
import scipy
import os
import datetime


def pad(im):
    maxi = np.max(im.shape)
    im_new = np.zeros((maxi, maxi, maxi))
    debut = (np.array((maxi, maxi, maxi))-np.array(im.shape))//2
    im_new[debut[0]:debut[0]+im.shape[0],
           debut[1]:debut[1]+im.shape[1],
           debut[2]:debut[2]+im.shape[2]] = im
    print(im_new.shape)
    return im_new


def pipeline(crop_file=None):
    if gv.reel:
        # crop_file = '/home/julin/Documents/imbilles/crops/0.2-1um_2/1um_0.037xy_0.05z_1/1.tif'
        Y = skio.imread(crop_file)
        gv.kernel_size = Y.shape
        print(gv.kernel_size)
        Y = Y/np.max(Y)
        observ_distri(Y, gv.resolution, 'Y distribution')
        gv.kernel_size = np.array(Y.shape)

    if gv.simulation:
        mutrue = np.array([0, 0, 0])
        Dtrue = gv.D
        print(Dtrue)
        Y, htrue, X = gen_observation(mutrue, Dtrue, sigma_noise=gv.sigma_noise)
        print(np.max(Y), np.min(Y), np.mean(Y))
        observ_distri(htrue, gv.resolution, 'h_true distribution')
        # print(Y)

    t = time.time()
    X = make_sphere(size = gv.kernel_size)
    # observ_distri(X, gv.resolution, 'X')
    observ(X, 0, 'X')
    observ(Y, 0, 'Y')
    #
    std = []
    for i in range(gv.kernel_size[2]):
        std.append(np.std(Y[:, :, i] - wiener(Y[:, :, i])))

    sigmaM = np.mean(std)
    print("sigmaM : ", sigmaM)
    # sigmaM = gv.sigma_noise
    # Chi2 loop
    if gv.lamba_loop:
        n_iter = 10
        ll_min = 0
        ll_max = 6
        N = Y.shape[0] * Y.shape[1] * Y.shape[2]
        lambdas = []
        errs = []
        temps = []

        for i in range(n_iter):
            ll = (ll_min + ll_max) / 2
            print()
            print(i)
            print("lambda : ", gv.lam)
            gv.lam = 10**ll
            t = time.time()
            D, mu, a, b, h_args, h, X = PSFEstimation(X, Y)
            temps.append(np.log(time.time() - t))
            chi2fit = np.linalg.norm(Y - a - b * fftconvolve(h, X, "same")) ** 2
            if chi2fit > sigmaM ** 2 * N:
                ll_max = ll
                print("ll goes down")
            else:
                print("ll goes up")
                ll_min = ll
            print("chi2 criteria : ", chi2fit / (sigmaM ** 2 * N))
            lambdas.append(ll)
            if gv.simulation:
                errs.append(np.linalg.norm(Dtrue - D) / np.linalg.norm(D))
                print("error : ", errs[-1])
                observ(htrue, 0, "htrue")

            observ(h_args, 0, "kargs")
            observ(h, 0, "k_est")
            observ(fftconvolve(h_args, X, "same"), 0, "k convolué avec X")
            observ_distri(h_args, gv.resolution, 'h_args distribution')

        if gv.simulation:
            plt.close("all")
            plt.plot(lambdas, errs, 'o')
            plt.title("Erreur relative sur D")
            plt.show()
        plt.plot(lambdas, temps, 'o')
        plt.title("Temps d'exécution (log10)")
        plt.show()
        print("SigmaM : ", sigmaM)
    chosen_D, chosen_mu, chosen_a, chosen_b, h_args, chosen_h, X = PSFEstimation(X, Y)
    temps_exec = time.time() - t

    if gv.simulation:
        observ(htrue, 0, "htrue")

    observ(h_args, 0, "kargs")
    observ(chosen_h, 0, "k_est")
    observ(fftconvolve(h_args, X, "same"), 0, "k convolué avec X")
    observ_distri(h_args, gv.resolution, 'h_args distribution')

    # Save
    if os.listdir(gv.save_path) == []:
        n = 0
    else:
        print(np.max(list(map(int, os.listdir(gv.save_path)))))
        n = np.max(list(map(int, os.listdir(gv.save_path)))) + 1

    gv.save_path = gv.save_path + str(n) + "/"
    os.mkdir(gv.save_path)
    os.mkdir(gv.save_path+"plots")
    os.mkdir(gv.save_path+"values")
    settings_file = open(gv.save_path+"settings.txt", "w")

    if gv.reel:
        # f"\nsigma_bruit = {gv.sigma_noise}" \
        # f"\nangle = {gv.angle}" \
        # f"\na_sim = {gv.a_sim}" \
        # f"\nb_sim = {gv.b_sim}" \

        sigma = np.linalg.inv(chosen_D)
        FWMH = np.sqrt(np.linalg.eig(sigma)[0])*gv.resolution*(2*np.sqrt(2*np.log(2)))
        txt = f"date = {datetime.datetime.now()}" \
              f"\nfile = {crop_file}" \
              f"\n\nresolution = {gv.resolution} " \
              f"\nfalse FWMH = {gv.FWMH} " \
              f"\ntrue FWMH = {FWMH} " \
              f"\nkernel_size = {gv.kernel_size})" \
              f"\n_lambda = {gv.lam}" \
              f"\ngam_h = {gv.gam_h}" \
              f"\nalpha = {gv.alpha}" \
              f"\ngamma = {gv.gamma}" \
              f"\ngam_mu = {gv.gam_mu}" \
              f"\ngam_D = {gv.gam_D}" \
              f"\ngam_a = {gv.gam_a}" \
              f"\ngam_b = {gv.gam_b}" \
              f"\nD_est = {chosen_D}" \
              f"\nm_uest = {chosen_mu}"\
              f"\na_est = {chosen_a}" \
              f"\nb_est = {chosen_b}" \
              f"\n\n temps execution : = {temps_exec}"

    if gv.simulation:
        txt =   f"date = {datetime.datetime.now()}" \
                f"\n\nresolution = {gv.resolution} " \
                f"\nFWMH = {gv.FWMH} " \
                f"\nangle = {gv.angle}" \
                f"\nkernel_size = {gv.kernel_size})" \
                f"\na_sim = {gv.a_sim}" \
                f"\nb_sim = {gv.b_sim}" \
                f"\n_lambda = {gv.lam}" \
                f"\ngam_h = {gv.gam_h}" \
                f"\nalpha = {gv.alpha}" \
                f"\ngamma = {gv.gamma}" \
                f"\ngam_mu = {gv.gam_mu}" \
                f"\ngam_D = {gv.gam_D}" \
                f"\ngam_a = {gv.gam_a}" \
                f"\ngam_b = {gv.gam_b}" \
                f"\nDtrue = {Dtrue}" \
                f"\nsigma_bruit = {gv.sigma_noise}" \
                f"\n\n temps execution : = {temps_exec}"

    settings_file.write(txt)
    settings_file.close()

    for i in range(len(gv.plots)):
        fig = gv.plots[i]
        fig.savefig(gv.save_path + "plots/" + gv.plot_names[i] + ".png")


    np.save(gv.save_path + "values/" + "hargs" + ".npy", h_args)
    np.save(gv.save_path + "values/" + "hest" + ".npy", chosen_h)

    np.save(gv.save_path + "values/" + "D" + ".npy", chosen_D)
    np.save(gv.save_path + "values/" + "X" + ".npy", X)
    np.save(gv.save_path + "values/" + "im" + ".npy", Y)
    np.save(gv.save_path + "values/" + "mu" + ".npy", chosen_mu)

    skio.imsave(gv.save_path + "values/" + "bille_simulee.tif", 1000 * X)
    skio.imsave(gv.save_path + "values/" + "PSFconvolueeIvrai.tif", fftconvolve(h_args, X, "same"))
    skio.imsave(gv.save_path + "values/" + "crop_bille.tif", Y)
    skio.imsave(gv.save_path + "values/" + "PSF_args.tif", h_args)

    if gv.simulation:
        np.save(gv.save_path + "values/" + "htrue" + ".npy", htrue)
        np.save(gv.save_path + "values/" + "Dtrue" + ".npy", Dtrue)

    # file = open(global_path + "tableau.txt", 'a')
    # file.write(gv.im_name.split('/')[0] + ',' + gv.im_name.split('/')[1] + ','+str(FWMH[0]) + ','+str(FWMH[1]) + ',' + str(FWMH[2]) + ',' + str(gv.reussi) + '\n')
    # file.close()

path_ims = '/home/julin/Documents/imbilles/crops/0.2-1um_2/'

# n_images = 0
# for ims in os.listdir(path_ims):
#     for crop in os.listdir(path_ims+ims+"/"):
#         n_images+=1

# i_image = 0
# global_path = "/home/julin/Documents/MPM_results/0.2-1um_3/"
# try:
#     os.mkdir(global_path)
# except:
#     pass

# file = open(global_path + "tableau.txt", 'a')
# file.write('set' + ',' + 'crop' + ',' + 'FWHM X' + ',' + 'FWHM Y'+ ',' + 'FWHM Z' + ',' + "convergence" + '\n')
# file.close()
# for ims in os.listdir(path_ims):
#     try:
#         os.mkdir(global_path+ims+"/")
#     except:
#         pass
#     for crop in os.listdir(path_ims+ims+"/"):
#         try:
#             os.mkdir(global_path+ims+"/crop"+crop[:-4])
#         except:
#             pass
#         i_image += 1
#         path_crop = path_ims + ims+"/" + crop
#         gv.im_name = ims+"/"+crop
#         print(path_crop)
#         print(gv.im_name)
#         print(f'Image {i_image}/{n_images}')
#         # pipeline(path_crop)
#         gv.save_path = global_path+ims+"/crop"+crop[:-4] + "/"
#         os.listdir(gv.save_path)
#         pipeline(path_crop)

pipeline("crops/Y/0.tif")

