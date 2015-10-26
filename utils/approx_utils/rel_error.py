import numpy as np


def rel_error(orig, approx, ord=None):
    return np.linalg.norm(orig - approx, ord=ord) / np.linalg.norm(orig, ord=ord)