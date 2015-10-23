# hmat
Integral equation solver based on hierarchical matrix approximation

Contains packages:
* black_box
* htree
* matrix_lowrank
* utils

## black_box
Package contains class BlackBox that emulates np.ndarray behaviour by function of calculating A[i, j] given

## htree
Package contains htree class takes BlackBox object, rank and block_size parameters, and return QuadTree-like structure with matvec method.

## matrix_lowrank
Package contains MatrixLowRank class that stores and manipulates with matrix $A$ in factor format $A=UV^T$
