import numpy as np


def r(query_matrix):
    return max(abs(row) for row in query_matrix)


def J(users_inputs):
    return np.amax(users_inputs) - np.amin(users_inputs)
