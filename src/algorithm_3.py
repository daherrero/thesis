from numpy.random import binomial, default_rng
from numpy import divide, multiply, identity, zeros
from scipy.stats import multivariate_normal


def R_RejSamp(A, v_i, sigma_sq, a, b, d):
    # Line 1
    a_v_i = A[:, v_i]

    # Line 2
    y_tilda_i = default_rng().multivariate_normal(zeros(d), multiply(sigma_sq, identity(d)))
    
    # Line 3
    covariance_matrix = multiply(sigma_sq, identity(d))
    f_a_v_i = multivariate_normal.pdf(y_tilda_i, mean=a_v_i, cov=covariance_matrix)
    f_0 = multivariate_normal.pdf(y_tilda_i, mean=zeros(d), cov=covariance_matrix)
    n_i = divide(divide(f_a_v_i, f_0), 2)

    # Lines 4 to 13
    if n_i >= a and n_i <= b:
        if binomial(1, n_i):
            return y_tilda_i
    return None
