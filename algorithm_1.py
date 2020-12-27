import numpy as np
import helpers


def R_i_Gauss(A, i, epsilon, delta):
    a_v_i = np.matmul(A, i)
    r = np.power(helpers.r(A), 2)
    sigma_sq = 2 * r * np.log(2/delta) / np.power(epsilon, 2)
    z_i = np.random.normal(0, np.multiply(sigma_sq, np.identity(A.shape[0])))
    return a_v_i+z_i
