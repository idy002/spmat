#include "csc.h"
#include "lsolve.h"


MatrixCSC lsolve_sparse_naive(MatrixCSC L, MatrixCSC b) {
    MatrixCSC b_dense = to_dense(b);
    lsolve_dense_naive(L, b_dense);
    return b_dense;
}
