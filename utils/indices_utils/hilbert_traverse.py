import numpy as np

from sklearn.neighbors import KDTree


def hilbert_traverse(size):
    x, y = np.mgrid[0:size, 0:size]
    data = list(zip(x.ravel(), y.ravel()))
    tree = KDTree(data, leaf_size=2)
    return np.array(tree.idx_array)