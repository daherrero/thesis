import numpy as np
import helpers
from algorithm_1 import R_i_Gauss


def Prot_Gauss(A, users_inputs, epsilon, delta):
    J = helpers.J(users_inputs)
    n = users_inputs.shape[0]
    d = A.shape[0]
    y_tilda = np.zeros(shape=(n, d))
    for user in range(n):
        y_tilda[user, ] = R_i_Gauss(A, users_inputs[user], epsilon, delta)
    y_bar = np.divide(np.sum(y_tilda, axis=0), n)
    if n < (np.power(d, 2)*np.log(2/epsilon))/(8*np.power(epsilon, 2)*np.log(J)):
        y_hat = ProjWSort(y_bar, A, 1)
        print("adios")
    else:
        y_hat = y_bar
        print("hola")
    return y_hat


def ProjWSort(y, w, a):
    w = w.sum(axis=1)
    z = np.zeros(w.shape[0])
    for i in range(w.shape[0]):
        z[i, ] = y[i, ]/w[i, ]
    z_perm = np.argsort(z)
    z = z[z_perm]
    sumWY = w[z_perm[0]] * y[z_perm[0]]
    Ws = w[z_perm[0]] * w[z_perm[0]]
    tau = (sumWY - a) / Ws
    i = 1
    while (i < w.shape[0]) and (z[i, ] > tau):
        sumWY += w[z_perm[i]] * y[z_perm[i]]
        Ws += w[z_perm[i]] * w[z_perm[i]]
        tau = (sumWY - a) / Ws
        i += 1
    x = np.zeros(w.shape[0])
    for i in range(w.shape[0]):
        if y[i] > w[i] * tau:
            x[i, ] = y[i] - w[i] * tau
        else:
            x[i, ] = 0
    return x