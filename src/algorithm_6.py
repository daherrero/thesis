import helpers
from numpy import multiply, divide, where, zeros
from numpy.random import default_rng
from time import time_ns

def Prot_AdSamp(D, epsilon, queries):
    # Timing and inferred parameters
    start_time_all = time_ns()
    r = helpers.l_infinity_r(queries)
    c_epsilon_r = multiply(helpers.c_epsilon(epsilon), r)
    
    # Line 1
    partition = default_rng().integers(queries.shape[0], size=D.shape[0])
    
    # Lines 2 to 8
    response_vector = zeros(queries.shape[0])
    start_time_queries = time_ns()
    for k in range(queries.shape[0]):
        # Get users with partition index k
        users_in_k = where(partition == k)[0]

        # Do nothing if no users in partition index
        if len(users_in_k) == 0:
            continue
        y_tilda_sum = 0

        # Lines 3 to 5
        for user in users_in_k:
            q_k_v_i = queries[k, D[user]]
            if random.rand() < (1/2)*(1+(divide(q_k_v_i, c_epsilon_r))):
                y_tilda_sum += c_epsilon_r
            else:
                y_tilda_sum -= c_epsilon_r

        # Line 6
        response_vector[k, ] = y_tilda_sum / len(users_in_k)
    
    # Timing functions
    end_time = time_ns()
    query_time_mean = divide(divide((end_time - start_time_queries), D.shape[0]), (10**6))
    total_time = divide((end_time - start_time_all), (10 ** 9))
    return (response_vector, query_time_mean, total_time)
