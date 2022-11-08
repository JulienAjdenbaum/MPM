import numpy as np


sphere_size = 1  # micrometers


# algorithm parameters
n_iter = 4000
stop_criteria = 1e-6
stop_criteria2 = 1e-4
print_n_iter = 20


cas = 'reel'

if cas=='reel':
    reel = True
else:
    reel = False
    simulation_simple = True
save_path = "/home/julin/Documents/MPM_results/"

if reel:
    resolution = np.array((0.05, 0.049, 0.049))
    plot = False
    _lambda = 1000
    gam_k = 1e-7
    alpha = gam_k
    gamma = 1
    gam_mu = gamma
    gam_D = gamma
    gam_a = gamma
    gam_b = gamma
    FWMH = np.array([0, 0, 0])


elif simulation_simple:

    resolution = np.array((0.05, 0.049, 0.049))
    FWMH = (75, 0.5, 0.5)
    angle = np.array([0, 0, 0])
    kernel_size = np.array((234, 44, 49))
    a_sim = 0
    b_sim = 1

    _lambda = 100
    gam_k = 1e-7
    alpha = gam_k
    gamma = 1
    gam_mu = gamma
    gam_D = gamma
    gam_a = gamma
    gam_b = gamma
    plot = True
    sigma_noise = 0.1

else:
    resolution = np.array((1, 1, 1))/5
    FWMH = np.array((5, 1, 1))
    angle = np.array([0, 0, 0])
    kernel_size = np.array((15, 15, 15))
    a_sim = 1
    b_sim = 2

    _lambda = 20
    gam_k = 1e-6
    alpha = gam_k
    gamma = 1
    gam_mu = gamma
    gam_D = gamma
    gam_a = gamma
    gam_b = gamma
    plot = False
    sigma_noise = 0.1


# data simulation parameters


interpolated = False

cnt = 0
debug = False

simulation = not reel

save_path = "/home/julin/Documents/MPM_results/"

plots = []
plot_names = []
im_name=""
reussi = False