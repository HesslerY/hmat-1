import numpy as np

from csvd import csvd
from tt.maxvol import maxvol
from matrix_lowrank import MatrixLowRank


def cross(a, r=None, niters=10):
    m, n = a.shape
    if r is None:
        r = min(m, n)
    r = min([m, n, r])
    indj = np.sort(np.random.choice(n, r, replace = False))

    for i in range(niters):
        C = a[:,indj]
        indi = np.sort(maxvol(C))
        R = a[indi, :]
        indj = np.sort(maxvol(R.T))

    C = np.linalg.solve(C[indi, :].T, C.T).T
    return C, R


def sum_factored(u1, v1, u2, v2, explicit_norm=False):
    assert u1.shape[0] == u2.shape[0]
    assert v1.shape[1] == v2.shape[1]
    assert u1.shape[1] == v1.shape[0]
    assert u2.shape[1] == v2.shape[0]
    r = max(u1.shape[1], u2.shape[1])

    U = np.hstack([u1, u2])
    V = np.vstack([v1, v2])
    QT, RT = np.linalg.qr(V.T)
    U = np.dot(U, RT.T)
    U, s, V = csvd(U, r)
    V = np.dot(V, QT.T)
    if explicit_norm:
        return U, s, V
    else:
        return U, np.dot(np.diag(s), V)


def low_rank_matrix_approx(A, r, delta=1e-8, maxiter=50):
    m, n = A.shape
    r = min(m, n, r)
    J = np.sort(np.random.choice(n, r, replace = False))
    approx_prev = MatrixLowRank((np.zeros((m, r)), np.zeros((r, n))), r)
    for i in range(maxiter):
        R = A[:, J]
        Q, T = np.linalg.qr(R)
        assert Q.shape == (m, r)
        I = np.sort(maxvol(Q))
        C = A[I, :].T
        assert C.shape == (n, r)
        Q, T = np.linalg.qr(C)
        assert Q.shape == (n, r)
        J = np.sort(maxvol(Q))
        QQ = Q[J, :]
        # We need to store the same as A matrix
        approx_next = MatrixLowRank((A[:, J], np.dot(Q, np.linalg.inv(QQ)).T), r=r)
        if (approx_next - approx_prev).norm < delta * approx_prev.norm:
            approx_next.round(delta)
            return approx_next.u, approx_next.norm * approx_next.v
        approx_prev = approx_next
    approx_prev.round(delta)
    return approx_prev.u, approx_prev.norm * approx_next.v
