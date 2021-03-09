import numpy as np
import helpers

def R_RejSamp(A, v_i, epsilon, n):
    a_v_i = np.matmul(A, v_i)
    r = np.power(helpers.r(A), 2)
    sigma_sq = 2 * r * np.log(2/2/np.power(n, 2)) / np.power(epsilon, 2)
    y_tilda_i = np.random.normal(0, np.multiply(sigma_sq, np.identity(A.shape[0])))
    f_a_v_i = 
    