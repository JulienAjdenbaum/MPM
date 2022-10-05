import numpy as np

# acquisition parameters
resolution = np.array((0.5, 0.049, 0.049))
sphere_size = 1  # micrometers

# algorithm parameters
_lambda = 1
gam_k = 1
alpha = 0.01
gam_mu = 1
gam_C = 1
plot = True

# data simulation parameters
FWMH = (1500, 0.5, 0.5)
interpolated = True