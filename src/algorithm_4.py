import numpy as np
import helpers
from algorithm_3 import R_RejSamp


def Prot_RejSamp(A, users_inputs, epsilon):
    J = helpers.J(users_inputs)
    n = users_inputs.shape[0]
    d = A.shape[0]

    r_sq = np.power(np.amax(np.linalg.norm(A, axis=1)), 2)
    sigma_sq = 4 * np.power(r_sq, 2) * np.divide(np.log(n), np.power(epsilon, 2))
    a = np.divide(np.exp(np.divide(-epsilon, 4)),2)
    b = np.divide(np.exp(np.divide(epsilon, 4)),2)

    y_tilda = np.zeros(shape=(n, d))
    n_hat = 0
    for user in range(n):
        r_i = R_RejSamp(A, users_inputs[user], epsilon, n, sigma_sq, a, b)
        if r_i is not None:
            y_tilda[n_hat, ] = r_i
            n_hat+=1

    y_bar = np.divide(np.sum(y_tilda, axis=0), n_hat)

    if n < (np.power(d, 2)*np.log(2))/(4*np.power(epsilon, 2)*np.log(J)):
        y_hat = helpers.ProjWSort(y_bar, A, 1)
    else:
        y_hat = y_bar
    return y_hat
