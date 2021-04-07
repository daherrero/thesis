import helpers, time
import numpy as np
from math import exp

def Prot_AdSamp(D, epsilon, queries):
    start_time_all = time.time_ns()
    partition = np.random.randint(queries.shape[0], size=D.shape[0])
    c_epsilon = helpers.c_epsilon(epsilon)
    r = helpers.linf_r(queries)
    c_epsilon_r = c_epsilon*r
    response_vector = np.zeros(queries.shape[0])
    start_time_queries = time.time_ns()
    for k in range(queries.shape[0]):
        users_in_k = np.where(partition == k)[0]
        if len(users_in_k) == 0:
            continue
        y_tilda_sum = 0
        for user in users_in_k:
            q_k_v_i = queries[k, D[user]]
            if np.random.rand() < (1/2)*(1+(q_k_v_i/(c_epsilon*r))):
                y_tilda_sum += c_epsilon_r
            else:
                y_tilda_sum -= c_epsilon_r
        response_vector[k, ] = y_tilda_sum / len(users_in_k)
    end_time = time.time_ns()
    query_time_mean = np.divide((end_time - start_time_queries), D.shape[0])
    total_time = np.divide((end_time - start_time_all), (10 ** 9))
    return (response_vector, query_time_mean, total_time)
