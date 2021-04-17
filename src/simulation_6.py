from algorithm_6 import Prot_AdSamp
from math import inf
import numpy as np
import sys, helpers


def simulation():
    if len(sys.argv) != 5:
        print("Usage: python3 simulation.py <categories> <queries> <users> <epsilon>")
        return
    else:
        J = int(sys.argv[1])
        d = int(sys.argv[2])
        n = int(sys.argv[3])
        epsilon = float(sys.argv[4])

    D = helpers.user_data_generator(n, J)
    A = helpers.linear_queries_generator(d, J)
    private_response, query_time, total_time = Prot_AdSamp(D, epsilon, A)
    linf_r = helpers.l_infinity_r(A)
    linf_error = helpers.prot_adsamp_estimated_error(epsilon, d, n, linf_r)
    real_response = helpers.realResponse(D, A)
    prot_adsamp_error = np.linalg.norm(x=(real_response-private_response), ord=inf)
    print(linf_error)
    print(prot_adsamp_error)
    print(private_response)
    print(real_response)
    print(query_time)
    print(total_time)

simulation()
