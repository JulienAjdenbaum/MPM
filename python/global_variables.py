import numpy as np

sphere_size = 1  # micrometers

# algorithm parameters
n_iter = 10000
stop_criteria = 1e-5
stop_criteria2 = 1e-4
print_n_iter = 1

lambda_loop = False
# 3 cas possible : 'reel', 'simple' (simulation simple), 'realiste" (simulation realiste)
cas = 'realiste'

if cas == 'reel':
    reel = True
    simulation_simple = None

elif cas == 'simple':
    reel = False
    simulation_simple = True

else:  # cas == 'sim_realiste':
    reel = False
    simulation_simple = False

save_path = "/home/julin/Documents/MPM_results/"

if reel:
    kernel_size = None
    resolution = np.array((0.05, 0.037, 0.037))
    plot = False
    lam = 100
    gam_h = None
    gamma = 1
    gam_mu = gamma
    gam_D = gamma
    gam_a = gamma
    gam_b = gamma
    FWMH = np.array([0, 0, 0])


elif not simulation_simple:
    resolution = np.array((0.05, 0.043, 0.043))
    D = [[0.02357253, 0.02227284, 0.06436205],
         [0.02227284, 0.20017209, 0.0174685],
         [0.06436205, 0.0174685, 0.21842065]]
    sigma = np.linalg.inv(D)
    FWMH = np.sqrt(np.linalg.eig(sigma)[0]) * resolution * (2 * np.sqrt(2 * np.log(2)))
    angle = np.array([0, 0, 0])
    kernel_size = np.array((93, 31, 43))
    a_sim = 0
    b_sim = 1

    lam = 3000
    gam_h = 1e-7
    gamma = 1
    gam_mu = gamma
    gam_D = gamma
    gam_a = gamma
    gam_b = gamma
    plot = False
    sigma_noise = 0

else:
    resolution = np.array((1, 1, 1)) / 5
    FWMH = np.array((5, 1, 1))
    angle = np.array([0, 0, 0])
    kernel_size = np.array((15, 15, 15))
    a_sim = 1
    b_sim = 2

    lam = 20
    gam_h = 1e-6
    gamma = 1
    gam_mu = gamma
    gam_D = gamma
    gam_a = gamma
    gam_b = gamma
    plot = False
    sigma_noise = 0.2

# data simulation parameters


interpolated = False

cnt = 0
debug = False

simulation = not reel

save_path = "/home/julin/Documents/MPM_results/"

plots = []
plot_names = []
im_name = ""
reussi = False
