# import numpy as np
# import plotly.graph_objects as go
#
# def get_value(idx, mu, C):
#     diff = np.diff([idx, mu])
#     return np.sqrt(np.linalg.det(C)/((2*np.pi)**3))*np.exp(-1/2*(diff)@C@(diff).T)
#
# mu = [0, 0, 0]
# theta = [0, 0, 0]
# KERNEL_SIGMA = 1
# Rx = np.array([[1, 0, 0],
#                [0, np.cos(theta[0]), -np.sin(theta[0])],
#                [0, np.sin(theta[0]), np.cos(theta[0])]])
# Ry = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
#                [0, 1, 0],
#                [-np.sin(theta[1]), 0, np.cos(theta[1])]])
# Rz = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
#
#                [np.sin(theta[2]), np.cos(theta[2]), 0],
#                [0, 0, 1]])
# R = Rx @ Ry @ Rz
# C = np.linalg.inv(R @ np.diag((np.array([KERNEL_SIGMA, KERNEL_SIGMA, KERNEL_SIGMA])).T) @ R.T)
#
# print(get_value([1,2,3], mu, C))
