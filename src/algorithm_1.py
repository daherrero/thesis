from numpy import zeros, identity, multiply
from numpy.random import default_rng

def R_i_Gauss(A, v_i, sigma_sq, d):
    # Real response
    a_v_i = A[:, v_i]

    # Gaussian noise
    z_i = default_rng().multivariate_normal(zeros(d), multiply(sigma_sq, identity(d)))

    return a_v_i+z_i
