import numpy as np
from math import exp


def Prot_AdSamp(D, epsilon, queries):
    print(D)
    print(queries)
    partition = np.random.randint(queries.shape[0], size=D.shape[0])
    c_epsilon = (exp(epsilon)+1) / (exp(epsilon)-1)
    r = np.amax(queries)
    c_epsilon_r = c_epsilon*r
    response_vector = np.zeros(queries.shape[0])
    for k in range(queries.shape[0]):
        # print(k)
        users_in_k = np.where(partition == k)[0]
        # print(users_in_k)
        if len(users_in_k) == 0:
            continue
        y_tilda_sum = 0
        for user in users_in_k:
            q_k_v_i = np.matmul(queries[k, ], D[user, ])
            # print("qkvi"+str(q_k_v_i))
            # print("prob1 "+str((1/2)*(1+(q_k_v_i/(c_epsilon*r)))))
            # print("prob2 "+str((1/2)*(1-(q_k_v_i/(c_epsilon*r)))))
            # print((1/2)*(1+(q_k_v_i/(c_epsilon*r)))+(1/2)*(1-(q_k_v_i/(c_epsilon*r))))
            # print("random"+str(np.random.rand()))
            if np.random.rand() < (1/2)*(1+(q_k_v_i/(c_epsilon*r))):
                y_tilda_sum += c_epsilon_r
            else:
                y_tilda_sum -= c_epsilon_r
                
        response_vector[k, ] = y_tilda_sum / len(users_in_k)
    return response_vector
