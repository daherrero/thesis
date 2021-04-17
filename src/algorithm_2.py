import helpers, time
from numpy import divide, power, log, zeros, sum
from algorithm_1 import R_i_Gauss


def Prot_Gauss(A, users_inputs, epsilon, delta, J):
    # Timing and inferred parameters
    start_time_all = time.time_ns()
    n = users_inputs.shape[0]
    d = A.shape[0]
    r_sq = power(helpers.l_2_r(A), 2)
    sigma_sq = 2 * r_sq * divide(log(2/delta), power(epsilon, 2))

    # Lines 1 to 3
    y_tilda = zeros(shape=(n, d))
    start_time_queries = time.time_ns()
    for user in range(n):
        y_tilda[user, ] = R_i_Gauss(A, users_inputs[user], sigma_sq, d)
    end_time_queries = time.time_ns()

    # Line 4
    y_bar = divide(sum(y_tilda, axis=0), n)

    # Lines 5 to 9
    if n < (power(d, 2)*log(2/epsilon))/(8*power(epsilon, 2)*log(J)):
        y_hat = helpers.ProjWSort(y_bar, A, 1)
    else:
        y_hat = y_bar

    # Timing functions
    end_time_all = time.time_ns()
    query_time_mean = divide(divide((end_time_queries - start_time_queries), n), (10**6))
    total_time = divide((end_time_all - start_time_all), (10 ** 9))

    return (y_hat, query_time_mean, total_time)
