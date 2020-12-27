import numpy as np
from math import exp


def Prot_AdSamp(D, epsilon, queries):
    partition = np.random.randint(queries.shape[0], size=D.shape[0])
    c_epsilon = (exp(epsilon)+1) / (exp(epsilon)-1)
    r = max(abs(row) for row in queries)
    response_vector = np.zeros(queries.shape[0])
    for k in range(queries.shape[0]):
        users_in_k = np.where(partition == k)[0]
        y_tilda_sum = 0
        for user in users_in_k:
            q_k_v_i = np.matmul(D[user, ], queries[k, ])
            if np.random.rand() < (1/2)*(1+(q_k_v_i/(c_epsilon*r))):
                y_tilda_sum += c_epsilon*r
            else:
                y_tilda_sum -= c_epsilon*r
        response_vector[k, ] = y_tilda_sum / users_in_k.len()
    return response_vector
