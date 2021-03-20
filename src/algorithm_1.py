import numpy as np


def R_i_Gauss(A, v_i, epsilon, delta, sigma_sq):
    d = A.shape[0]
    a_v_i = A[:, v_i]
    z_i = np.random.multivariate_normal(np.zeros(d), np.multiply(sigma_sq, np.identity(d)))
    return a_v_i+z_i
