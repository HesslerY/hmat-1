import numpy as np

from utils.approx_utils import csvd, tsvd

class MatrixLowRank(object):
    def __init__(self, data, r=None):
        if type(data) == np.ndarray:
            if len(data.shape) != 2:
                raise ValueError('supports only 2d arrays')
            self.r = min(data.shape) if r is None else r
            self.u, s, self.v = csvd(data, self.r)
            self.norm = np.linalg.norm(s)
            s /= self.norm
            self.v = np.dot(np.diag(s), self.v)
        if type(data) == tuple:
            if len(data) == 2:
                u, v = data
                # we have u, v matrices
                if (u.shape[1] != v.shape[0]):
                    raise ValueError('u, v must be multiplicable, but have shapes {}, {}'.format(u.shape, v.shape))
                self.r = u.shape[1] if r is None else r
                qt, rt = np.linalg.qr(v.T)
                u = np.dot(u, rt.T)
                self.u, s, v = csvd(u, self.r)
                self.norm = np.linalg.norm(s)
                if self.norm == 0:
                    self.v = np.zeros_like(qt.T)
                    return
                s /= self.norm
                self.v = np.dot(np.diag(s), np.dot(v, qt.T))
            else:
                raise ValueError('supports only a = uv matrix factorization')

    def __add__(self, other):
        if type(other) != MatrixLowRank:
            raise ValueError('second summand must be matrix_lowrank')
        r = self.r + other.r
        return MatrixLowRank((np.hstack([self.u, other.u]), np.vstack([self.norm * self.v, other.norm * other.v])), r)

    def __radd__(self, other):
        if type(other) != MatrixLowRank:
            raise ValueError('second summand must be matrix_lowrank')
        return other.__sum__(self)

    def __sub__(self, other):
        if type(other) != MatrixLowRank:
            raise ValueError('second summand must be matrix_lowrank')
        r = self.r + other.r
        return MatrixLowRank((np.hstack([self.u, other.u]), np.vstack([self.norm * self.v, -other.norm * other.v])), r)

    def __rsub__(self, other):
        if type(other) != MatrixLowRank:
            raise ValueError('second summand must be matrix_lowrank')
        return other.__sub__(self)

    def round(self, eps=1e-8):
        qt, rt = np.linalg.qr(self.v.T)
        u = np.dot(self.u, rt.T)
        self.u, s, v = tsvd(u, delta=np.linalg.norm(u)*eps)
        self.r = s.size
        self.norm *= np.linalg.norm(s)
        s /= np.linalg.norm(s)
        self.v = np.dot(np.diag(s), np.dot(v, qt.T))

    def full_matrix(self):
        return np.dot(self.u, self.norm * self.v)