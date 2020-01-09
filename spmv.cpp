#include "csc.h"
#include "spmv.h"
#include <cassert>

MatrixCSC spmv_dense(MatrixCSC A, MatrixCSC b) {
    assert(b.m == b.nnz && b.n == 1);
    int n = A.n;
    MatrixCSC c = csc_zeros(n, 1);
    for(int i = 0; i < n; i++) {
        for(int k = A.c[i]; k < A.c[i+1]; k++) {
            int j = A.r[k];
            c.v[j] += A.v[k] * b.v[i];
        }
    }
    return c;
}

