import numpy as np
from tt.maxvol import maxvol

def cross(a, r=None, niters=10):
    m, n = a.shape
    if r is None:
        r = min(m, n)
    indj = np.sort(np.random.choice(n, r, replace = False))

    for i in range(niters):
        C = a[:,indj]
        indi = np.sort(maxvol(C))
        R = a[indi, :]
        indj = np.sort(maxvol(R.T))

    C = np.linalg.solve(C[indi, :].T, C.T).T
    return np.dot(C, R)