#include "csc.h"
#include "lsolve.h"
#include "spmv.h"
#include <cstdio>
#include <chrono>
#include <omp.h>

void test_lsolve_dense_naive(MatrixCSC L, MatrixCSC b) {
    MatrixCSC x = csc_copy(b);
    auto t1 = std::chrono::high_resolution_clock::now();
    lsolve_dense_naive(L, x);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_elapsed = t2 - t1;
    MatrixCSC c = spmv_dense(L, x);
    printf("dense rhs, naive, elapsed time: %f ms\n", time_elapsed.count() * 1000.0);
    printf("MAE: %.15f\n", csc_mae(b, c));
    save_mm("./data/dense_b_naive_x.mtx", x);
    save_mm("./data/dense_b_naive_c.mtx", c);
    x.free();
    c.free();
}

void test_lsolve_dense_opt(MatrixCSC L, MatrixCSC b, int win) {
    Graph g = build_graph(L);
    CGraph cg = compress_graph(g);
    Partitions p = greedy_partition(g, win, omp_get_max_threads());
//    p.summary();
    MatrixCSC x = csc_copy(b);
    auto t1 = std::chrono::high_resolution_clock::now();
    lsolve_dense_opt(L, x, cg, p);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_elapsed = t2 - t1;
    MatrixCSC c = spmv_dense(L, x);
    printf("dense rhs, optimized, elapsed time: %f ms\n", time_elapsed.count() * 1000.0);
    printf("MAE: %.15f\n", csc_mae(b, c));
    save_mm("./data/dense_b_opt_x.mtx", b);
    p.free();
    cg.free();
    x.free();
    c.free();
}

map<int, double> enumerate_win(MatrixCSC L, MatrixCSC b, int begin, int end, int step) {
    Graph g = build_graph(L);
    CGraph cg = compress_graph(g);
    map<int, double> result;
    for(int win = begin; win < end; win += step) {
        Partitions p = greedy_partition(g, 5, omp_get_max_threads());
        MatrixCSC x = csc_copy(b);
        auto t1 = std::chrono::high_resolution_clock::now();
//        double t1 = omp_get_wtime();
        lsolve_dense_opt(L, x, cg, p);
        auto t2 = std::chrono::high_resolution_clock::now();
//        double t2 = omp_get_wtime();
        std::chrono::duration<double> time_elapsed = t2 - t1;
        result[win] = time_elapsed.count() * 1000.0;
//        result[win] = t2 - t1;
        x.free();
    }
    return result;
}

void test_lsolve_sparse_naive(MatrixCSC L, MatrixCSC b) {
    auto t1 = std::chrono::high_resolution_clock::now();
    MatrixCSC x = lsolve_sparse_naive(L, b);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_elapsed = t2 - t1;
    MatrixCSC c = spmv_dense(L, x);
    printf("sparse rhs, naive, elapsed time: %f ms\n", time_elapsed.count() * 1000.0);
    printf("MAE: %.15f\n", csc_mae(b, c));
    save_mm("./data/sparse_b_naive_x.mtx", b);
    x.free();
    c.free();
}

void test_omp() {
    int nthread, tid;
    {
        tid = omp_get_thread_num();
        printf("Hello world from thread %d\n", tid);
        if(tid == 0) {
            nthread = omp_get_num_threads();
            printf("Number of threads = %d\n", nthread);
        }
    }
}

int main() {
    printf("reading matrices...");
    MatrixCSC L = read_mm("./data/af_0_k101/af_0_k101.mtx");
    MatrixCSC b_sparse = read_mm("./data/af_0_k101/b_sparse_af_0_k101.mtx");
    MatrixCSC b_dense = read_mm("./data/af_0_k101/b_dense_af_0_k101.mtx");
//    MatrixCSC L = read_mm("./data/sample/L.mtx");
//    MatrixCSC b_dense = read_mm("./data/sample/b_dense.mtx");
//    MatrixCSC b_sparse = read_mm("./data/sample/b_sparse.mtx");
    printf(" done\n");

    printf("max num threads: %d\n", omp_get_max_threads());

    test_lsolve_dense_naive(L, b_dense);
    printf("\n");
    test_lsolve_dense_opt(L, b_dense, 5);
    auto result = enumerate_win(L, b_dense, 1, 21, 1);
    printf("\n");
    printf("enumerate different win:\n");
    for(auto pr : result)
        printf("%d: %.6f\n", pr.first, pr.second);
}

