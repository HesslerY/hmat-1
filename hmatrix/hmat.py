import numpy as np

from black_box import BlackBox
from utils import distance_matrix

from utils.approx_utils import cross, low_rank_matrix_approx, rel_error
from matrix_lowrank import MatrixLowRank
from utils.indices_utils import hilbert_traverse, break_ranges
from utils.matrix_utils import conjugate, conjugate_sparse

from scipy.sparse.linalg import LinearOperator


class hmat_node(object):
    def __init__(self, tree, ranges):
        self.tree = tree
        self.ranges = ranges
        self.low_rank = False

        xlen, ylen = ranges[0][1] - ranges[0][0], ranges[1][1] - ranges[1][0]

        if (xlen <= tree.leaf_side or ylen <= tree.leaf_side):
            # leaf block~---~terminate partiotion
            self.mat = tree.mat[slice(*ranges[0]), slice(*ranges[1])]
            self.is_leaf = True
        elif (tree.pattern[slice(*ranges[0]), slice(*ranges[1])].nnz == 0):
            # there is no close-relations between points - we start
            # low-rank approximation
            self.u, self.v = low_rank_matrix_approx(tree.mat[slice(*ranges[0]), slice(*ranges[1])], r=self.tree.r)
            self.low_rank = True
            self.is_leaf = True
        else:
            # non-leaf node - continue recursion
            self.is_leaf = False
            ranges_nw, ranges_ne, ranges_sw, ranges_se = break_ranges(ranges)
            self.nw = hmat_node(tree, ranges_nw)
            self.ne = hmat_node(tree, ranges_ne)
            self.sw = hmat_node(tree, ranges_sw)
            self.se = hmat_node(tree, ranges_se)
        return

    def full_part(self, mat):
        if (self.is_leaf and self.low_rank):
            mat[slice(*self.ranges[0]), slice(*self.ranges[1])] = np.dot(self.u, self.v)
        elif (self.is_leaf):
            mat[slice(*self.ranges[0]), slice(*self.ranges[1])] = self.mat
        else:
            self.nw.full_part(mat)
            self.ne.full_part(mat)
            self.sw.full_part(mat)
            self.se.full_part(mat)
        return

    def check_part(self, mat):
        print("ranges:{}, leaf?{}, lr?{}".format(self.ranges, self.is_leaf, self.low_rank))
        if (self.is_leaf and self.low_rank):
            print("rel_error:{}".format(rel_error(mat[slice(*self.ranges[0]), slice(*self.ranges[1])],
                                                  np.dot(self.u, self.v))))
            print("-"*80)
        elif (self.is_leaf):
            print("rel_error:{}".format(rel_error(mat[slice(*self.ranges[0]), slice(*self.ranges[1])],
                                                  self.mat)))
            print("-"*80)
        else:
            self.nw.check_part(mat)
            self.ne.check_part(mat)
            self.sw.check_part(mat)
            self.se.check_part(mat)
        return

    def matvec_part(self, vec, result):
        if (self.is_leaf and self.low_rank):
            result[slice(*self.ranges[0])] += np.dot(self.u, np.dot(self.v, vec[slice(*self.ranges[1])]))
        elif (self.is_leaf):
            result[slice(*self.ranges[0])] += np.dot(self.mat, vec[slice(*self.ranges[1])])
        else:
            self.nw.matvec_part(vec, result)
            self.ne.matvec_part(vec, result)
            self.sw.matvec_part(vec, result)
            self.se.matvec_part(vec, result)
        return

    def rmatvec_part(self, vec, result):
        if (self.is_leaf and self.low_rank):
            result[slice(*self.ranges[1])] += np.dot(self.v.conj().T, np.dot(self.u.conj().T, vec[slice(*self.ranges[0])]))
        elif (self.is_leaf):
            result[slice(*self.ranges[1])] += np.dot(self.mat.conj().T, vec[slice(*self.ranges[0])])
        else:
            self.nw.rmatvec_part(vec, result)
            self.ne.rmatvec_part(vec, result)
            self.sw.rmatvec_part(vec, result)
            self.se.rmatvec_part(vec, result)
        return

    def count_params_part(self):
        if (self.is_leaf and self.low_rank):
            return self.u.size + self.v.size
        elif (self.is_leaf):
            return self.mat.size
        else:
            return self.nw.count_params_part() + \
                self.ne.count_params_part() + \
                self.sw.count_params_part() + \
                self.se.count_params_part()


class hmat(LinearOperator):
    def __init__(self, mat, r=10, leaf_side=16):
        LinearOperator.__init__(self, dtype=mat.dtype, shape=mat.shape,
                                matvec=self._matvec, rmatvec=self._rmatvec)
        self.mat = BlackBox(mat)
        self.r = r
        self.leaf_side = leaf_side
        self.leaf_size = leaf_side**2

        N = int(np.sqrt(self.mat.shape[0]))

        self.pattern = distance_matrix(N, format='coo')
        self.pattern = self.pattern.tocsr()
        perm = hilbert_traverse(N)
        conjugate_sparse(self.pattern, perm)
        self.mat.permutate(perm)
        self.root = hmat_node(self, tuple(zip((0, 0), self.mat.shape)))
        return

    def full_matrix(self):
        mat = np.zeros(self.mat.shape)
        self.root.full_part(mat)
        return mat

    def _matvec(self, vec):
        assert len(vec.shape) == 1, 'vec must be vector'
        assert self.mat.shape[1] == vec.shape[0],\
            'mat of shape {shp}, cannot matvec on vec of shape {vshp}'.format(shp=self.mat.shape, vshp=vec.shape)
        result = np.zeros(self.mat.shape[0])
        self.root.matvec_part(vec, result)

    def _rmatvec(self, vec):
        assert len(vec.shape) == 1, 'vec must be vector'
        assert self.mat.shape[0] == vec.shape[0],\
            'mat of shape {shp}, cannot matvec on vec of shape {vshp}'.format(shp=self.mat.shape, vshp=vec.shape)
        result = np.zeros(self.mat.shape[1])
        self.root.rmatvec_part(vec, result)

    def count_params(self):
        return self.root.count_params_part()

    def check(self, orig):
        self.root.check_part(orig)