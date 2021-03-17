import numpy as np


def R_i_Gauss(A, v_i, epsilon, delta, r):
    a_v_i = np.matmul(A, v_i)
    r_sq = np.power(r, 2)
    sigma_sq = 2 * r * np.log(2/delta) / np.power(epsilon, 2)
    z_i = np.random.multivariate_normal(np.zeros(A.shape[0]), np.multiply(sigma_sq, np.identity(A.shape[0])))
    return a_v_i+z_i
