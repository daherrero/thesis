import numpy as np


def algorithm_6(D, epsilon, queries):
    partition = np.random.randint(queries.shape[0], size=D.shape[0])
    for k in range(queries.shape[0]):
        k_query = queries[k, ]
        for user in np.where(partition == k)[0]:
