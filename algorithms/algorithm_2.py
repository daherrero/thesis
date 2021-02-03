import numpy as np
import helpers
from algorithm_1 import R_i_Gauss


def Prot_Gauss(A, users_inputs, epsilon, delta):
    J = helpers.J(users_inputs)
    n = users_inputs.shape[0]
    d = A.shape[0]
    y_tilda = np.zeros(n, d)
    for user in range(n):
        y_tilda[user] = R_i_Gauss[users_inputs[user]]
    y_bar = np.divide(np.sum(y_tilda, axis=1), n)
    if n < (np.power(d, 2)*np.log(2/epsilon))/(8*np.power(epsilon, 2)*np.log(J)):
        # y_hat = np.argmin
        return
    else:
        y_hat = y_bar
    return y_hat
