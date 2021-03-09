import numpy as np


def R_i_Gauss(A, v_i, epsilon, delta):
    a_v_i = np.matmul(A, v_i)
    r = np.power(np.amax(A), 2)
    sigma_sq = 2 * r * np.log(2/delta) / np.power(epsilon, 2)
    z_i = np.random.normal(0, np.multiply(sigma_sq, np.identity(A.shape[0]))).sum(axis=1)
    return a_v_i+z_i
