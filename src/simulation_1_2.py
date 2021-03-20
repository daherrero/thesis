from algorithm_2 import Prot_Gauss
from math import exp, sqrt, log
import numpy as np
import sys, helpers


def simulation():
    if len(sys.argv) != 6:
        print("Usage: python3 simulation.py <categories> <queries> <users> <epsilon> <delta>")
        return
    else:
        J = int(sys.argv[1])
        d = int(sys.argv[2])
        n = int(sys.argv[3])
        epsilon = float(sys.argv[4])
        delta = float(sys.argv[5])

    D = helpers.user_data_generator(n, J)
    A = helpers.linear_queries_generator(d, J)
    print(Prot_Gauss(A, D, epsilon, delta, J))
    print(helpers.realResponse(D, A))


simulation()
