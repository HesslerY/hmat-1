import numpy as np


def break_ranges(ranges):
    xbegin = ranges[0][0]
    xend = ranges[0][1]
    ybegin = ranges[1][0]
    yend = ranges[1][1]
    xlen = xend - xbegin
    ylen = yend - ybegin
    ranges_nw = ((xbegin, xbegin + xlen // 2), (ybegin, ybegin + ylen // 2))
    ranges_ne = ((xbegin, xbegin + xlen // 2), (ybegin + ylen // 2, yend))
    ranges_sw = ((xbegin + xlen // 2, xend), (ybegin, ybegin + ylen // 2))
    ranges_se = ((xbegin + xlen // 2, xend), (ybegin + ylen // 2, yend))
    return ranges_nw, ranges_ne,\
           ranges_sw, ranges_se

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