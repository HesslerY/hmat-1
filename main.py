import numpy as np
import timeit

from hmatrix import hmat
from black_box import BlackBox
from utils.approx_utils import rel_error
from utils.matrix_utils import conjugate

import cProfile

from matplotlib import pyplot as plt


def gen_func(N):
    c = 1. / N**2
    def func(indices):
        with np.errstate(divide='ignore', invalid='ignore'):
            res = c * 1. / np.abs(indices[:, 0] - indices[:, 1])
            res[res == np.inf] = 0
        return res
    return func

def gen_mat(N):
    c = 1. / N**2
    indices = np.arange(N**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = c * 1. / np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
        res[res == np.inf] = 0
    return res


import scipy.sparse as scisp
import scipy.sparse.linalg


def main_func(N=64):
    # create fbox and hmat from our function
    fbox = BlackBox(gen_func(N), shape=(N**2, N**2))
    hm = hmat(fbox, r=3, leaf_side=16)
    #orig = gen_mat(N)
    # get original matrix
    #approx = hm.full_matrix()
    #print('approximation accuracy: {}'.format(rel_error(orig, approx)))
    #print hm.count_params(), int(np.prod(orig.shape))

    x = np.ones((N**2))

    #rel_err_list_full = []
    rel_err_list_hmat = []

    x_hmat = scipy.sparse.linalg.cg(hm, hm.matvec(x), x0=None, tol=1e-6, maxiter=50,
                                    callback=lambda xk: rel_err_list_hmat.append(rel_error(x, xk)))
    #x_full = scipy.sparse.linalg.cg(fbox[:, :], np.dot(fbox[:, :], x), x0=None, tol=1e-6, maxiter=50,
    #                                callback=lambda xk: rel_err_list_full.append(rel_error(x, xk)))

    #print(rel_error(np.dot(fbox[:, :], x), hm.matvec(x)))
    #print(x_full[1], x_hmat[1])
    #print(rel_error(x, x_full[0]), rel_error(x, x_hmat[0]))
    #for res, res_n in zip(rel_err_list_full, rel_err_list_hmat):
    #    print(res, res_n)
    print(x_hmat[1])
    print('rel_error: {}'.format(rel_error(x, x_hmat[0])))
    for res in rel_err_list_hmat:
        print(res)

    hmat_params = hm.count_params()
    full_params = int(np.prod(fbox.shape))
    print('hmat params: {}, full params: {}'.format(hmat_params, full_params))
    print('Compression_rate: {}'.format(1. - 1.*hmat_params/full_params))
    return


def matvec_test(r=3):
    ns = np.arange(10, 32, 1)
    times_full = []
    times_hmat = []
    for n in ns:
        fbox = BlackBox(gen_func(n), shape=(n**2, n**2))
        hm = hmat(fbox, r=r, leaf_side=16)
        x = np.ones((n**2))
        orig = gen_mat(n)
        times_full.append(timeit.timeit(lambda: np.dot(orig, x)))
        times_hmat.append(timeit.timeit(lambda: hm.matvec(x)))
    plt.plot(ns**2, times_full, 'r', ns**2, times_hmat, 'g')
    plt.show()
    return times_full, times_hmat


if __name__ == "__main__":
    #main_func(128)
    cProfile.run('main_func(64)')
    #matvec_test(r=3)