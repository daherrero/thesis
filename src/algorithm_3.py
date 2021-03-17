import numpy as np
from scipy.stats import multivariate_normal
import helpers


def R_RejSamp(A, v_i, epsilon, n, r):
    a_v_i = np.matmul(A, v_i)
    d = A.shape[0]
    r_sq = np.power(r, 2)
    
    sigma_sq = 4 * np.power(r_sq, 2) * np.divide(np.log(n), np.power(epsilon, 2))
    y_tilda_i = np.random.default_rng().multivariate_normal(np.zeros(d), np.multiply(sigma_sq, np.identity(d)))

    f_a_v_i = multivariate_normal.pdf(y_tilda_i, mean=a_v_i, cov=np.multiply(sigma_sq, np.identity(d)))
    f_0 = multivariate_normal.pdf(y_tilda_i, mean=np.zeros(d), cov=np.multiply(sigma_sq, np.identity(d)))

    n_i = np.divide(np.divide(f_a_v_i, f_0), 2)
    a = np.exp(np.divide(np.divide(-epsilon, 4), 2))
    b = np.exp(np.divide(np.divide(epsilon, 4), 2))

    if n_i >= a and n_i <= b:
        if np.binomial(1, n_i):
            return y_tilda_i
    return None
