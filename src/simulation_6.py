from algorithm_6 import Prot_AdSamp

from numpy.linalg import norm
from math import inf

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

    # Queries matrix and user data generation
    D = helpers.user_data_generator(n, J)
    A = helpers.linear_queries_generator(d, J)

    # Responses
    private_response, time_per_query, total_time = Prot_AdSamp(D, epsilon, A)
    real_response = helpers.realResponse(D, A)

    # Errors
    estimated_error = helpers.prot_adsamp_estimated_error(epsilon, d, n, helpers.l_infinity_r(A))
    real_error = norm(x=(real_response-private_response), ord=inf)

    # Printing
    print(f"Private response: {private_response}\nReal response: {real_response}")
    print(f"Estimated error: {estimated_error}\nReal error: {real_error}")
    print(f"Time per query (s): {time_per_query}\nTotal time (ms) {total_time}")

simulation()
