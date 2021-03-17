from algorithm_2 import Prot_Gauss
from math import exp, sqrt, log
import numpy as np
import sys


def data_generator_normal(n, J, loc=0, scale=1):
    return np.random.normal(loc, scale, size=(n, J))


def linear_queries_generator(d, J):
    return np.random.rand(d, J)


def trueResponse(D, A):
    y_tilda = np.zeros(shape=(D.shape[0], A.shape[0]))
    for user in range(D.shape[0]):
        y_tilda[user, ] = np.matmul(A, D[user])
    y_sum = y_tilda.sum(axis=0)
    y = np.divide(y_sum, D.shape[0])
    return y


def simulation():
    if len(sys.argv) != 6:
        print("Usage: python3 simulation.py <variables> <queries> <users> <epsilon> <delta>")
        return
    else:
        J = int(sys.argv[1])
        d = int(sys.argv[2])
        n = int(sys.argv[3])
        epsilon = float(sys.argv[4])
        delta = float(sys.argv[5])

    D = data_generator_normal(n, J)
    A = linear_queries_generator(d, J)
    print(Prot_Gauss(A, D, epsilon, delta))
    print(trueResponse(D, A))


simulation()
