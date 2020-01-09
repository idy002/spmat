#include "csc.h"
#include "lsolve.h"
#include <cassert>

void lsolve_dense_naive(MatrixCSC L, MatrixCSC b) {
    assert(b.m == b.nnz && b.n == 1); // assert dense b
    int n = L.n;
    int *r = L.r;
    int *c = L.c;
    dtype *bv = b.v;
    dtype *Lv = L.v;
    for(int i = 0; i < n; i++) {
        bv[i] /= Lv[c[i]];
        for(int k = c[i] + 1; k < c[i+1]; k++) {
            bv[r[k]] -= bv[i] * Lv[k];
        }
    }
}


