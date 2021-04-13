from math import e, inf
from algorithm_2 import Prot_Gauss
from algorithm_4 import Prot_RejSamp
from algorithm_6 import Prot_AdSamp

import numpy as np
import helpers, csv, time

def simulation(categories, queries, users, epsilon, delta):
    J = categories
    d = queries
    n = users
    D = helpers.user_data_generator(n, J)
    A = 50*helpers.linear_queries_generator(d, J)

    l2_r = helpers.l2_r(A)
    linf_r = helpers.linf_r(A)

    l2_gauss_error = helpers.prot_gauss_estimated_error(epsilon, delta, d, l2_r, n, J) 
    l2_error = helpers.prot_rejsamp_estimated_error(epsilon, d, l2_r, n, J)
    linf_error = helpers.prot_adsamp_estimated_error(epsilon, d, n, linf_r)

    start_real_response = time.time_ns()
    real_response = helpers.realResponse(D, A)
    end_real_response = time.time_ns()
    total_real_time = np.divide((end_real_response - start_real_response), (10**9))
    query_real_time = np.divide(np.divide((end_real_response - start_real_response), users), (10**6))

    prot_gauss_response, gauss_query_time, gauss_total_time = Prot_Gauss(A, D, epsilon, delta, J)
    prot_gauss_error = np.linalg.norm(real_response-prot_gauss_response)

    prot_rejsamp_response, rejsamp_query_time, rejsamp_total_time = Prot_RejSamp(A, D, epsilon, J)
    prot_rejsamp_error = np.linalg.norm(real_response-prot_rejsamp_response)

    prot_adsamp_reponse, adsamp_query_time, adsamp_total_time = Prot_AdSamp(D, epsilon, A)
    prot_adsamp_error_linf = np.linalg.norm(x=(real_response-prot_adsamp_reponse), ord=inf)
    prot_adsamp_error_l2 = np.linalg.norm(real_response-prot_adsamp_reponse)

    print(f"Epsilon: {epsilon}, Delta: {delta}, N_cat: {categories}, users: {users}")
    print(f"Real response = {real_response}")
    print(f"Estimated Gauss error: {l2_gauss_error}")
    print(f"ProtGauss error: {prot_gauss_error}, response = {prot_gauss_response}")
    print(f"Estimated RejSamp error: {l2_error}")
    print(f"ProtRejSamp error: {prot_rejsamp_error}, response = {prot_rejsamp_response}")
    print(f"Estimated AdSamp error: {linf_error}")
    print(f"ProtAdSamp error: {prot_adsamp_error_linf}, response = {prot_adsamp_reponse}")
    print(f"ProtAdSamp l2-error: {prot_adsamp_error_l2}\n")

    with open('results_with_time.csv', mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(['ProtGauss', epsilon, delta, J, d, n, prot_gauss_response, l2_gauss_error, prot_gauss_error, gauss_query_time, gauss_total_time])
        results_writer.writerow(['ProtRejSamp', epsilon, "NA", J, d, n, prot_rejsamp_response, l2_error, prot_rejsamp_error, rejsamp_query_time, rejsamp_total_time])
        results_writer.writerow(['ProtAdSamp', epsilon, delta, J, d, n, prot_adsamp_reponse, linf_error, prot_adsamp_error_linf, adsamp_query_time, adsamp_total_time])
        results_writer.writerow(['Real', 'NA', 'NA', J, d, n, real_response, "NA", "NA", query_real_time, total_real_time])

categories = [50]
queries = [5]
users = [100000]
epsilons = [1]
deltas = [1]

for n_users in users:
    for epsilon in epsilons:
        for delta in deltas:
            for category in categories:
                for query in queries:
                    simulation(category, query, n_users, epsilon, np.multiply(delta, np.divide(1,n_users)))