import numpy as np

# acquisition parameters
resolution = np.array((0.1, 0.1, 0.1))
sphere_size = 1  # micrometers

# algorithm parameters
_lambda = 1
gam_k = 1
alpha = 0.01
gam_mu = 1
gam_C = 1
plot = False

# data simulation parameters
FWMH = (1.5, 0.5, 0.5)
interpolated = False