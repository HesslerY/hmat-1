import numpy as np

from utils.approx_utils import cross, rel_error

if __name__ == "__main__":
    A = np.arange(10)[:, np.newaxis] + np.arange(10)[np.newaxis, :]
    print(A)
    B = cross(A, r=2)
    print(rel_error(A, B))