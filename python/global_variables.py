import numpy as np

# acquisition parameters
resolution = np.array((0.5, 0.049, 0.049))
# resolution = np.array((1, 1, 1))/2
sphere_size = 1  # micrometers

# algorithm parameters
_lambda = 1
gam_k = 0.0001
alpha = gam_k
gam_mu = 1
gam_D = 1
gam_a = 1
gam_b = 1
plot = True

# data simulation parameters
FWMH = (1500, 0.5, 0.5)
# FWMH = (10, 1, 1)
a_sim = 0
b_sim = 1
kernel_size = np.array((234, 44, 49))
interpolated = True

cnt = 0
debug = False

reel = False
simulation = not reel