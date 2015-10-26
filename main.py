import numpy as np

from hmatrix import hmat
from black_box import BlackBox
from utils.approx_utils import rel_error
from utils.matrix_utils import conjugate


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


if __name__ == "__main__":
    # create fbox and hmat from our function
    N = 16
    fbox = BlackBox(gen_func(N), shape=(N**2, N**2))
    hm = hmat(fbox, r=20, leaf_side=16)
    #orig = gen_mat(N)
    # get original matrix
    #approx = hm.full_matrix()
    #print('approximation accuracy: {}'.format(rel_error(orig, approx)))
    #print hm.count_params(), int(np.prod(orig.shape))

    x = np.ones((N**2))

    rel_err_list_full = []
    rel_err_list_hmat = []

    x_full = scipy.sparse.linalg.cg(hm, hm.matvec(x), x0=None, tol=1e-6, maxiter=50,
                                    callback=lambda xk: rel_err_list_full.append(rel_error(x, xk)))
    x_hmat = scipy.sparse.linalg.cg(fbox[:, :], np.dot(fbox[:, :], x), x0=None, tol=1e-6, maxiter=50,
                                    callback=lambda xk: rel_err_list_hmat.append(rel_error(x, xk)))

    print(rel_error(np.dot(fbox[:, :], x), hm.matvec(x)))
    print(x_full[1], x_hmat[1])
    print(rel_error(x, x_full[0]), rel_error(x, x_hmat[0]))
    for res, res_n in zip(rel_err_list_full, rel_err_list_hmat):
        print(res, res_n)