import numpy as np


def indices_from_shape(shape):
    return tuple((np.arange(dim) for dim in shape))


def break_indices(indices):
    indices = np.asarray(indices)
    m, n = indices[0].size, indices[1].size
    return ((indices[0][:m//2], indices[1][:n//2]), (indices[0][:m//2], indices[1][n//2:])),\
            ((indices[0][m//2:], indices[1][:n//2]), (indices[0][m//2:], indices[1][n//2:]))


def indices_unveil(indices):
    indices = np.asarray(indices)
    return np.vstack([np.repeat(indices[0], indices[1].size),
               np.tile(indices[1], indices[0].size)]).T