#include "csc.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cassert>

struct Triple {
    int x;
    int y;
    dtype v;
    Triple(int x, int y, dtype v):x(x), y(y), v(v){}
};
bool operator<(const Triple &lhs, const Triple &rhs) {
    return lhs.y < rhs.y || (lhs.y == rhs.y && lhs.x < rhs.x);
}

MatrixCSC from_triples(int m, int n, const std::vector<Triple> &triples) {
    MatrixCSC mat;
    int nnz = triples.size();
    mat.m = m;
    mat.n = n;
    mat.nnz = nnz;
    mat.r = new int[nnz];
    mat.c = new int[m + 1];
    mat.v = new dtype[nnz];
    int col_index = 0;
    mat.c[0] = 0;
    for(int i = 0; i < nnz; i++) {
        mat.r[i] = triples[i].x;
        mat.v[i] = triples[i].v;
        if(col_index < triples[i].y) {
            for(int yy = col_index + 1; yy <= triples[i].y; yy++)
                mat.c[yy] = i;
            col_index = triples[i].y;
        }
    }
    for(int yy = col_index + 1; yy <= n; yy++)
        mat.c[yy] = nnz;
    return mat;
}

MatrixCSC read_mm(const char *fpath) {
    auto f = fopen(fpath, "rt");
    char object[100], format[100], field[100], symmetry[100];
    fscanf(f, "%*s %s %s %s %s\n", object, format, field, symmetry);
    char buf[1000];
    do
        fgets(buf, sizeof(buf), f);
    while(buf[0] == '%');
    std::vector<Triple> triples;

    int m, n, nnz;
    if(strcmp(format, "coordinate") == 0) {
        sscanf(buf, "%d %d %d", &m, &n, &nnz);
        for(int i = 0; i < nnz; i++) {
            int x, y;
            double v;
            fscanf(f, "%d %d %lf", &x, &y, &v);
            triples.emplace_back(x-1, y-1, v);
        }
    } else {
        sscanf(buf, "%d %d", &m, &n);
        nnz = m * n;
        for(int i = 0; i < nnz; i++) {
            double v;
            fscanf(f, "%lf", &v);
            triples.emplace_back(i % m, i / m, v);
        }
    }
    std::sort(triples.begin(), triples.end());
    return from_triples(m, n, triples);
}

MatrixCSC to_dense(MatrixCSC mat) {
    int m = mat.m;
    int n = mat.n;
    std::vector<dtype> full(n * m, 0.0);
    for(int i = 0; i < n; i++) {
        for(int k = mat.c[i]; k < mat.c[i+1]; k++) {
            int j = mat.r[k];
            full[i * m + j] = mat.v[k];
        }
    }
    std::vector<Triple> triples;
    for(int k = 0; k < n * m; k++) {
        int x = k % m;
        int y = k / m;
        triples.emplace_back(x, y, full[k]);
    }
    sort(triples.begin(), triples.end());
    return from_triples(m, n, triples);
}

void MatrixCSC::free() {
    delete [] c;
    delete [] r;
    delete [] v;
}

void csc_print(MatrixCSC mat) {
    printf("rows: %d\n", mat.m);
    printf("cols: %d\n", mat.n);
    printf("nonzeros: %d\n", mat.nnz);
    printf("values:\n");
    for(int i = 0; i < mat.nnz; i++)
        printf("%4.2f ", mat.v[i]);
    printf("\n");
    printf("row indices:\n");
    for(int i = 0; i < mat.nnz; i++)
        printf("%4d ", mat.r[i]);
    printf("\n");
    printf("columns indices:\n");
    for(int i = 0; i < mat.n + 1; i++)
        printf("%d ", mat.c[i]);
    printf("\n");
}

void save_mm(const char *fpath, MatrixCSC mat) {
    auto f = fopen(fpath, "wt");
    fprintf(f, "%%%%MatrixMarket matrix coordinate real symmetric\n");
    int m = mat.m;
    int n = mat.n;
    fprintf(f, "%d %d %d\n", m, n, mat.nnz);
    for(int i = 0; i < n; i++) {
        for(int k = mat.c[i]; k < mat.c[i+1]; k++) {
            int j = mat.r[k];
            fprintf(f, "%d %d %.20lf\n", j, i, (double)mat.v[k]);
        }
    }
}

MatrixCSC csc_zeros(int m, int n) {
    MatrixCSC mat;
    int nnz = m * n;
    mat.m = m;
    mat.n = n;
    mat.nnz = m * n;
    mat.r = new int[nnz];
    mat.c = new int[n + 1];
    mat.v = new dtype[mat.nnz];
    for(int i = 0; i < n; i++)
        mat.c[i] = i * m;
    mat.c[n] = nnz;
    for(int i = 0; i < nnz; i++) {
        mat.r[i] = i % m;
        mat.v[i] = 0.0;
    }
    return mat;
}

MatrixCSC csc_copy(MatrixCSC mat) {
    MatrixCSC copyed;
    copyed.m = mat.m;
    copyed.n = mat.n;
    copyed.nnz = mat.nnz;
    copyed.r = new int[copyed.nnz];
    copyed.c = new int[copyed.n + 1];
    copyed.v = new dtype[copyed.nnz];
    memcpy(copyed.r, mat.r, sizeof(int) * copyed.nnz);
    memcpy(copyed.c, mat.c, sizeof(int) * (copyed.n + 1));
    memcpy(copyed.v, mat.v, sizeof(dtype) * copyed.nnz);
    return copyed;
}

dtype csc_mae(MatrixCSC a, MatrixCSC b) {
    a = to_dense(a);
    b = to_dense(b);
    dtype sum = 0.0;
    assert(a.nnz == b.nnz);
    int nnz = a.nnz;
    for(int i = 0; i < nnz; i++)
        sum += std::abs(a.v[i] - b.v[i]);
    a.free();
    b.free();
    return sum / (dtype)nnz;
}
