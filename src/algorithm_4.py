import numpy as np
import helpers, time
from algorithm_3 import R_RejSamp


def Prot_RejSamp(A, users_inputs, epsilon, J):
    start_time_all = time.time_ns()
    n = users_inputs.shape[0]
    d = A.shape[0]
    r_sq = np.power(helpers.l2_r(A), 2)

    sigma_sq = 4 * r_sq * np.divide(np.log(n), np.power(epsilon, 2))

    a = np.divide(np.exp(np.divide(-epsilon, 4)),2)
    b = np.divide(np.exp(np.divide(epsilon, 4)),2)
    
    y_tilda = np.zeros(shape=(n, d))
    n_hat = 0
    start_time_queries = time.time_ns()
    for user in range(n):
        r_i = R_RejSamp(A, users_inputs[user], sigma_sq, a, b)
        if r_i is not None:
            y_tilda[n_hat, ] = r_i
            n_hat+=1
    end_time_queries = time.time_ns()

    y_bar = np.divide(np.sum(y_tilda, axis=0), n_hat)

    if n < (np.power(d, 2)*np.log(2))/(4*np.power(epsilon, 2)*np.log(J)):
        print("hola")
        y_hat = helpers.ProjWSort(y_bar, A, 1)
    else:
        y_hat = y_bar

    end_time_all = time.time_ns()
    query_time_mean = np.divide(np.divide((end_time_queries - start_time_queries), n), (10**6))
    total_time = np.divide((end_time_all - start_time_all), (10 ** 9))
    return (y_hat, query_time_mean, total_time)
