__author__ = 'cnst'


import numpy as np
import scipy as sp

from scipy import sparse


def conjugate_sparse(mat, permutation):
    mat_type = type(mat)
    mat_coo = sp.sparse.coo_matrix(mat)
    idr = np.argsort(permutation)
    mat_coo.row = permutation[mat_coo.row]
    mat_coo.col = permutation[mat_coo.col]
    return mat_type(mat_coo)


def conjugate(A, permutation, block_size=None):
    assert len(A.shape) == 2, 'A must be matrix'
    assert A.shape[0] == A.shape[1], 'A must be square matrix, but has shape {}'.format(A.shape)
    if block_size is not None:
        bs = A.shape[0] // permutation.size
        permutation = np.repeat(permutation, bs)*bs + np.tile(np.arange(bs), permutation.size)
    A[:, :] = A[permutation, :]
    A[:, :] = A[:, permutation]