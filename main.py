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
    xn = x[hm.mat.perm[np.arange(x.size)]]

    res_list = []
    res_list_n = []

    xrn = scipy.sparse.linalg.cg(hm, hm.matvec(xn), x0=None, tol=1e-6, maxiter=50, callback=lambda xk: res_list_n.append(rel_error(xn, xk)))
    xr = scipy.sparse.linalg.cg(fbox[:, :], np.dot(fbox[:, :], x), x0=None, tol=1e-6, maxiter=50, callback=lambda xk: res_list.append(rel_error(x, xk)))

    print(rel_error(np.dot(fbox[:, :], x), hm.matvec(xn)[hm.mat.perm.argsort()[np.arange(x.size)]]))
    print(xr[1], xrn[1])
    print(rel_error(x, xr[0]), rel_error(xn, xrn[0]))
    for res, res_n in zip(res_list, res_list_n):
        print(res, res_n)