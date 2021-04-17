import numpy as np
import helpers, time
from algorithm_1 import R_i_Gauss


def Prot_Gauss(A, users_inputs, epsilon, delta, J):
    start_time_all = time.time_ns()
    n = users_inputs.shape[0]
    d = A.shape[0]
    
    r_sq = np.power(helpers.l_2_r(A), 2)
    sigma_sq = 2 * r_sq * np.divide(np.log(2/delta), np.power(epsilon, 2))

    y_tilda = np.zeros(shape=(n, d))
    start_time_queries = time.time_ns()
    for user in range(n):
        y_tilda[user, ] = R_i_Gauss(A, users_inputs[user], epsilon, delta, sigma_sq)
    end_time_queries = time.time_ns()
    y_bar = np.divide(np.sum(y_tilda, axis=0), n)
    if n < (np.power(d, 2)*np.log(2/epsilon))/(8*np.power(epsilon, 2)*np.log(J)):
        y_hat = helpers.ProjWSort(y_bar, A, 1)
    else:
        y_hat = y_bar
    end_time_all = time.time_ns()
    query_time_mean = np.divide(np.divide((end_time_queries - start_time_queries), n), (10**6))
    total_time = np.divide((end_time_all - start_time_all), (10 ** 9))
    return (y_hat, query_time_mean, total_time)
