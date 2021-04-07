from math import e, inf
from algorithm_2 import Prot_Gauss
from algorithm_4 import Prot_RejSamp
from algorithm_6 import Prot_AdSamp

import numpy as np
import helpers, csv

def simulation(categories, queries, users, epsilon, delta):
    J = categories
    d = queries
    n = users
    D = helpers.user_data_generator(n, J)
    A = helpers.linear_queries_generator(d, J)

    l2_r = helpers.l2_r(A)
    linf_r = helpers.linf_r(A)

    l2_gauss_error = helpers.l2_gaussian_error(epsilon, delta, d, l2_r, n, J) 
    l2_error = helpers.l2_error(epsilon, d, l2_r, n, J)
    linf_error = helpers.linf_error(epsilon, d, n, linf_r)

    real_response = helpers.realResponse(D, A)

    prot_gauss_response = Prot_Gauss(A, D, epsilon, delta, J)
    prot_gauss_error = np.linalg.norm(real_response-prot_gauss_response)

    prot_rejsamp_response = Prot_RejSamp(A, D, epsilon, J)
    prot_rejsamp_error = np.linalg.norm(real_response-prot_rejsamp_response)

    prot_adsamp_reponse, _, _ = Prot_AdSamp(D, epsilon, A)
    prot_adsamp_error = np.linalg.norm(x=(real_response-prot_adsamp_reponse), ord=inf)
    
    print(l2_gauss_error)
    print(prot_gauss_error)
    print(l2_error)
    print(prot_rejsamp_error)
    print(linf_error)
    print(prot_adsamp_error)

    with open('results.csv', mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(['ProtGauss', epsilon, delta, J, d, n, prot_gauss_response, l2_gauss_error, prot_gauss_error])
        results_writer.writerow(['ProtRejSamp', epsilon, "NA", J, d, n, prot_rejsamp_response, l2_error, prot_rejsamp_error])
        results_writer.writerow(['ProtAdSamp', epsilon, delta, J, d, n, prot_adsamp_reponse, l2_error, prot_adsamp_error])
        results_writer.writerow(['Real', 'NA', 'NA', J, d, n, real_response, "NA", "NA"])

categories = 50
queries = 5
users = [1000000]
epsilons = [0.01, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
deltas = [1]

for i in users:
    for j in epsilons:
        for k in deltas:
            simulation(categories, queries, i, j, np.multiply(k, np.divide(1,i)))