//
// Created by yaoyao on 2020/1/8.
//

#ifndef SPMAT_LSOLVE_H
#define SPMAT_LSOLVE_H
#include "csc.h"
#include <map>
#include <vector>
#include <set>

using std::map;
using std::set;
using std::vector;

struct Graph {
    int n;
    map<int, set<int>> e;
    void adde(int u, int v) {
        e[u].insert(v);
    }
};

struct CGraph {
    int n;
    int* head;
    int* e;

    void free() {
        delete [] head;
        delete [] e;
    }
};

struct Partitions {
    int num_phase;
    int* phase_workers;
    int** lens;
    int*** nodes;

    void init(vector<vector<vector<int>>> &part) {
        num_phase = part.size();
        phase_workers = new int[num_phase];
        lens = new int*[num_phase];
        nodes = new int**[num_phase];
        for(int i = 0; i < num_phase; i++) {
            phase_workers[i] = part[i].size();
            lens[i] = new int[phase_workers[i]];
            nodes[i] = new int*[phase_workers[i]];
            for(int j = 0; j < phase_workers[i]; j++) {
                lens[i][j] = part[i][j].size();
                nodes[i][j] = new int[lens[i][j]];
                for(int k = 0; k < lens[i][j]; k++)
                    nodes[i][j][k] = part[i][j][k];
            }
        }
    }

    void free() {
        for(int i = 0; i < num_phase; i++) {
            for(int j = 0; j < phase_workers[i]; j++) {
                delete [] nodes[i][j];
            }
            delete[] nodes[i];
            delete[] lens[i];
        }
        delete [] nodes;
        delete [] phase_workers;
    }
    void summary() {
        printf("phases: %d\n", num_phase);
        printf("phase workers: \n");
        for(int i = 0; i < num_phase; i++) {
            for(int j = 0; j < phase_workers[i]; j++)
                printf("%d ", lens[i][j]);
            printf("\n");
        }
//        printf("\n");
    }
};

void lsolve_dense_naive(MatrixCSC L, MatrixCSC b);

Graph build_graph(MatrixCSC L);
CGraph compress_graph(Graph &g);
Partitions naive_partition(Graph &g);
Partitions greedy_partition(Graph &g, int win, int workers);
void lsolve_dense_opt(MatrixCSC L, MatrixCSC b, CGraph g, Partitions p);

MatrixCSC lsolve_sparse_naive(MatrixCSC L, MatrixCSC b);

#endif //SPMAT_LSOLVE_H
