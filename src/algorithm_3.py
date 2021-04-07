import numpy as np
from scipy.stats import multivariate_normal


def R_RejSamp(A, v_i, sigma_sq, a, b):
    a_v_i = A[:, v_i]
    d = A.shape[0]

    y_tilda_i = np.random.default_rng().multivariate_normal(np.zeros(d), np.multiply(sigma_sq, np.identity(d)))
    
    f_a_v_i = multivariate_normal.pdf(y_tilda_i, mean=a_v_i, cov=np.multiply(sigma_sq, np.identity(d)))
    f_0 = multivariate_normal.pdf(y_tilda_i, mean=np.zeros(d), cov=np.multiply(sigma_sq, np.identity(d)))

    n_i = np.divide(np.divide(f_a_v_i, f_0), 2)
    if n_i >= a and n_i <= b:
        if np.random.binomial(1, n_i):
            return y_tilda_i
    return None
