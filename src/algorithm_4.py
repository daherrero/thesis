import helpers
from numpy import divide, exp, power, log, sum, zeros
from time import time_ns
from algorithm_3 import R_RejSamp


def Prot_RejSamp(A, users_inputs, epsilon, J):
    # Timing and inferred parameters
    start_time_all = time_ns()
    n = users_inputs.shape[0]
    d = A.shape[0]
    r_sq = power(helpers.l_2_r(A), 2)
    sigma_sq = 4 * r_sq * divide(log(n), power(epsilon, 2))
    # Algorithm 3 line 4 range (a,b)
    a = divide(exp(divide(-epsilon, 4)),2)
    b = divide(exp(divide(epsilon, 4)),2)
    

    # Lines 1 to 4
    # Response vector
    y_tilda = zeros(shape=(n, d))
    # N valid responses
    n_hat = 0
    start_time_queries = time_ns()
    for user in range(n):
        r_i = R_RejSamp(A, users_inputs[user], sigma_sq, a, b, d)
        if r_i is not None:
            y_tilda[n_hat, ] = r_i
            n_hat+=1
    end_time_queries = time_ns()

    # Line 5
    y_bar = divide(sum(y_tilda, axis=0), n_hat)

    # Lines 6 to 10
    if n < (power(d, 2)*log(2))/(4*power(epsilon, 2)*log(J)):
        print("hola")
        y_hat = helpers.ProjWSort(y_bar, A, 1)
    else:
        y_hat = y_bar

    # Timing functions
    end_time_all = time_ns()
    query_time_mean = divide(divide((end_time_queries - start_time_queries), n), (10**6))
    total_time = divide((end_time_all - start_time_all), (10 ** 9))
    
    return (y_hat, query_time_mean, total_time)
