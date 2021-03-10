import numpy as np
from scipy.stats import norm
import helpers


def R_RejSamp(A, v_i, epsilon, n):
    a_v_i = np.matmul(A, v_i)

    r = np.power(helpers.r(A), 2)
    sigma_sq = 4 * np.power(r, 2) * np.log(n) / np.power(epsilon, 2)
    y_tilda_i = np.random.normal(0, np.multiply(sigma_sq, np.identity(A.shape[0]))).sum(axis=1)

    f_a_v_i = norm.pdf(y_tilda_i, np.mean(a_v_i), np.sqrt(sigma_sq))
    f_0 = norm.pdf(y_tilda_i, 0, np.sqrt(sigma_sq))
    n_i = np.multiply(np.divide(np.divide(f_a_v_i, f_0)), 2)

    a = np.exp(np.divide(np.divide(-epsilon, 4), 2))
    b = np.exp(np.divide(np.divide(epsilon, 4), 2))
    if n_i >= a and n_i <= b:
        if np.binomial(1, n_i):
            return y_tilda_i
    return None
