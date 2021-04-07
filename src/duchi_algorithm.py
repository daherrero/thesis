from math import pi
import numpy as np
import scipy


def B(alpha, d, r):
    a = np.divide(np.add(np.exp(alpha), 1), np.add(np.exp(alpha), 1))
    b_1 = np.multiply(d, np.sqrt(pi), scipy.special.gamma(np.add(np.divide(np.subtract(d, 1), 2), 1)))
    b_2 = scipy.special.gamma(np.add(np.divide(d,2), 1))
    return np.multiply(r, a, np.divide(b_1, b_2))


def z_cand(v_tilda, B, d):
    candidate = np.random.default_rng().multivariate_normal(np.zeros(d), np.identity(d))
    return np.multiply(candidate, np.divide(np.exp(B, np.divide(1, d)), np.linalg.norm(candidate)))


def strategy26(v, r, alpha, B, d):
    d = v.shape[0]
    b = B(alpha, d, r)
    v_norm = np.linalg.norm(v)
    if np.random.rand() < np.add(0.5, np.divide(v_norm, np.multiply(2, r))):
        v_tilda = np.divide(np.multiply(r, v), v_norm)
    else:
        v_tilda = -np.divide(np.multiply(r, v), v_norm)
    
    z_candidate = z_cand(v_tilda, b, d)

    pi_alpha = np.divide(np.exp(alpha), np.add(np.exp(alpha), 1))
    if np.random.binomial(1, pi_alpha):
        while np.inner(z_candidate, v_tilda) <= 0:
            z_candidate = z_cand(v_tilda, b, d)
    else:
        while np.inner(z_candidate, v_tilda) > 0:
            z_candidate = z_cand(v_tilda, b, d)
    
    return z_candidate


def duchi_algorithm(A, users_inputs, epsilon, J):
