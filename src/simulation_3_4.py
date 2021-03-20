from algorithm_4 import Prot_RejSamp
from math import exp, sqrt, log
import numpy as np
import sys, helpers


def simulation():
    if len(sys.argv) != 5:
        print("Usage: python3 simulation.py <variables> <queries> <users> <epsilon>")
        return
    else:
        J = int(sys.argv[1])
        d = int(sys.argv[2])
        n = int(sys.argv[3])
        epsilon = float(sys.argv[4])

    D = helpers.user_data_generator(n, J)
    A = helpers.linear_queries_generator(d, J)
    print(Prot_RejSamp(A, D, epsilon, J))
    print(helpers.realResponse(D, A))


simulation()
