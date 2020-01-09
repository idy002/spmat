#ifndef SPMAT_CSC_H
#define SPMAT_CSC_H

typedef float dtype;

struct MatrixCSC {
    int m;      // number of rows
    int n;      // number of columns
    int nnz;    // number of non-zeros
    int *c;     // indices of columns in v, r
    int *r;     // indices of rows in matrix
    dtype *v;   // values

    void free();
};

MatrixCSC read_mm(const char *fpath);
MatrixCSC to_dense(MatrixCSC mat);
MatrixCSC csc_zeros(int m, int n);
MatrixCSC csc_copy(MatrixCSC mat);
dtype csc_mae(MatrixCSC a, MatrixCSC b);
void csc_print(MatrixCSC mat);
void save_mm(const char *fpath, MatrixCSC mat);

#endif //SPMAT_CSC_H
