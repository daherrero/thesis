import numpy as np

def realResponse(D, A):
    y_tilda = np.zeros(shape=(D.shape[0], A.shape[0]))
    for user in range(D.shape[0]):
        y_tilda[user, ] = A[:, D[user]]
    y_sum = y_tilda.sum(axis=0)
    y = np.divide(y_sum, D.shape[0])
    return y


# Data generation
def user_data_generator(n, J, loc=0, scale=1):
    return np.random.default_rng().integers(J, size=n)


def linear_queries_generator(d, J):
    return np.random.default_rng().random(size=(d, J))


# Inferred parameters
def c_epsilon(epsilon):
    return np.divide((np.exp(epsilon)+1), (np.exp(epsilon)-1))


def J(users_inputs):
    return np.amax(users_inputs) - np.amin(users_inputs)


def l_infinity_r(query_matrix):
    return np.amax(query_matrix)


def l_2_r(query_matrix):
    return np.amax(np.linalg.norm(query_matrix, axis=0))


# Upper bound estimated errors
def prot_adsamp_estimated_error(epsilon, d, n, r):
    return np.multiply(r, np.sqrt(np.divide(np.multiply(np.power(c_epsilon(epsilon), 2), np.multiply(d, np.log(d))), n)))


def prot_rejsamp_estimated_error(epsilon, d, r, n, J):
    a = np.power(np.divide(np.multiply(280*np.log(J),np.log(n)), np.multiply(n, np.power(epsilon, 2))), np.divide(1,4))
    b = np.sqrt(np.divide(np.multiply(10*d,np.log(n)), np.multiply(n, np.power(epsilon, 2))))
    print(a)
    print(b)
    return np.multiply(r, np.minimum(a, b))


def prot_gauss_estimated_error(epsilon, delta, d, r, n, J):
    a = np.power(np.divide(np.multiply(32*np.log(J),np.log(np.divide(2, delta))), np.multiply(n, np.power(epsilon, 2))), np.divide(1,4))
    b = np.sqrt(np.divide(np.multiply(2*d,np.log(np.divide(2, delta))), np.multiply(n, np.power(epsilon, 2))))
    return np.multiply(r, np.minimum(a, b))


# L1-ball projection
def ProjWSort(y, w, a):
    w = w.sum(axis=1)
    z = np.zeros(w.shape[0])
    for i in range(w.shape[0]):
        z[i, ] = y[i, ]/w[i, ]
    z_perm = np.argsort(z)
    z = z[z_perm]
    sumWY = w[z_perm[0]] * y[z_perm[0]]
    Ws = w[z_perm[0]] * w[z_perm[0]]
    tau = (sumWY - a) / Ws
    i = 1
    while (i < w.shape[0]) and (z[i, ] > tau):
        sumWY += w[z_perm[i]] * y[z_perm[i]]
        Ws += w[z_perm[i]] * w[z_perm[i]]
        tau = (sumWY - a) / Ws
        i += 1
    x = np.zeros(w.shape[0])
    for i in range(w.shape[0]):
        if y[i] > w[i] * tau:
            x[i, ] = y[i] - w[i] * tau
        else:
            x[i, ] = 0
    return x

