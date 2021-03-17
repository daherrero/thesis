import numpy as np
import helpers
from algorithm_1 import R_i_Gauss


def Prot_Gauss(A, users_inputs, epsilon, delta):
    J = helpers.J(users_inputs)
    n = users_inputs.shape[0]
    d = A.shape[0]
    r_sq = np.power(np.amax(np.linalg.norm(A, axis=1)), 2)
    sigma_sq = 2 * r_sq * np.log(2/delta) / np.power(epsilon, 2)

    y_tilda = np.zeros(shape=(n, d))
    for user in range(n):
        y_tilda[user, ] = R_i_Gauss(A, users_inputs[user], epsilon, delta, sigma_sq)
    y_bar = np.divide(np.sum(y_tilda, axis=0), n)
    if n < (np.power(d, 2)*np.log(2/epsilon))/(8*np.power(epsilon, 2)*np.log(J)):
        y_hat = helpers.ProjWSort(y_bar, A, 1)
    else:
        y_hat = y_bar
    return y_hat
