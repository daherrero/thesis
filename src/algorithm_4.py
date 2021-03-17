import numpy as np
import helpers
from algorithm_3 import R_RejSamp


def Prot_RejSamp(A, users_inputs, epsilon):
    J = helpers.J(users_inputs)
    n = users_inputs.shape[0]
    d = A.shape[0]
    r = np.amax(np.linalg.norm(A, axis=1))

    y_tilda = np.zeros(shape=(n, d))
    n_hat = 0
    for user in range(n):
        r_i = R_RejSamp(A, users_inputs[user], epsilon, n, r)
        if r_i:
            y_tilda[n_hat, ] = r_i
            n_hat+=1

    y_bar = np.divide(np.sum(y_tilda, axis=0), n_hat)

    if n < (np.power(d, 2)*np.log(2))/(4*np.power(epsilon, 2)*np.log(J)):
        y_hat = helpers.ProjWSort(y_bar, A, 1)
    else:
        y_hat = y_bar
    return y_hat
