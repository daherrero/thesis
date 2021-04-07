from algorithm_4 import Prot_RejSamp
from math import exp, sqrt, log
import numpy as np
import sys, helpers


def simulation():
    if len(sys.argv) != 5:
        print("Usage: python3 simulation.py <categories> <queries> <users>> <epsilon>")
        return
    else:
        J = int(sys.argv[1])
        d = int(sys.argv[2])
        n = int(sys.argv[3])
        epsilon = float(sys.argv[4])

    D = helpers.user_data_generator(n, J)
    A = helpers.linear_queries_generator(d, J)
    private_response = Prot_RejSamp(A, D, epsilon, J)
    real_response = helpers.realResponse(D, A)
    print(private_response)
    print(real_response)
    print(helpers.l2_error(epsilon, d, helpers.l2_r(A), n, J))
    print(np.linalg.norm(real_response-private_response))

simulation()
