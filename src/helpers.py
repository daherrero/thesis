import numpy as np
import math

def realResponse(D, A):
    y_tilda = np.zeros(shape=(D.shape[0], A.shape[0]))
    for user in range(D.shape[0]):
        y_tilda[user, ] = A[:, D[user]]
    y_sum = y_tilda.sum(axis=0)
    y = np.divide(y_sum, D.shape[0])
    return y

def trueResponse(D, A):
    y_tilda = np.zeros(shape=(D.shape[0], A.shape[0]))
    for user in range(D.shape[0]):
        y_tilda[user, ] = np.matmul(A, D[user])
    y_sum = y_tilda.sum(axis=0)
    y = np.divide(y_sum, D.shape[0])
    return y

def user_data_generator(n, J, loc=0, scale=1):
    return np.random.randint(0, J, size=n)

def data_generator_normal(n, J, loc=0, scale=1):
    return np.random.normal(loc, scale, size=(n, J))

def linear_queries_generator(d, J):
    return np.random.rand(d, J)

def r(query_matrix):
    return max(abs(row) for row in query_matrix)

def J(users_inputs):
    return np.amax(users_inputs) - np.amin(users_inputs)

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


"""
from https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246
Module to compute projections on the positive simplex or the L1-ball

A positive simplex is a set X = {\mathbf{x} | \sum_i x_i = s, x_i \geq 0 }

The (unit) L1-ball is the set X = { \mathbf{x} | || x ||_1 \leq 1 }

Adrien Gaidon - INRIA - 2011
"""


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.

    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / rho
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the L1-ball

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s

    Notes
    -----
    Solves the problem by a reduction to the positive simplex case

    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w
