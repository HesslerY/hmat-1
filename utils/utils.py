import numpy as np
import scipy as sp

from scipy import sparse


def rel_error(orig, approx, ord=None):
    return np.linalg.norm(orig - approx, ord=ord) / np.linalg.norm(approx, ord=ord)


def distance_matrix(block_size, format='coo'):
    ex = np.ones(block_size)
    T = sp.sparse.spdiags(np.vstack([ex, ex, ex]),
                          [1, 0, -1], block_size, block_size, format=format)
    return sp.sparse.kron(T, T, format=format)
