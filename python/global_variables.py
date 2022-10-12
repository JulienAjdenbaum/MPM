import numpy as np

reel = False

# acquisition parameters

# resolution = np.array((0.05, 0.049, 0.049))
# FWMH = (75, 0.5, 0.5)
# angle = np.array([0, 0, 0])
# kernel_size = np.array((234, 44, 49))
# a_sim = 0
# b_sim = 1
#
# _lambda = 1
# gam_k = 1e-7
# alpha = gam_k
# gamma = 1
# gam_mu = gamma
# gam_D = gamma
# gam_a = gamma
# gam_b = gamma
# plot = True



resolution = np.array((1, 1, 1))/10
FWMH = np.array((5, 1, 1))
angle = np.array([np.pi / 4, 0, -np.pi / 6])
kernel_size = np.array((31, 31, 31))
a_sim = 1
b_sim = 2

_lambda = 1
gam_k = 1e-6
alpha = gam_k
gamma = 1e-2
gam_mu = gamma
gam_D = gamma
gam_a = gamma
gam_b = gamma
plot = True

sphere_size = 1  # micrometers

# algorithm parameters
n_iter = 10000


# data simulation parameters


interpolated = True

cnt = 0
debug = False

simulation = not reel