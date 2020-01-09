#include "lsolve.h"
#include "csc.h"
#include <cassert>
#include <vector>
#include <omp.h>
#include <queue>
#include <algorithm>
#include <cstdlib>

using std::vector;
using std::queue;

struct DisjointSet {
    int n;
    vector<int> fa;

    DisjointSet(int n_):n(n_) {
        fa.resize(n);
        init();
    }
    void init() {
        for(int i = 0; i < n; i++)
            fa[i] = i;
    }
    int find(int u) {
        return fa[u] == u ? u : (fa[u] = find(fa[u]));
    }
    bool union_sets(int u, int v) {
        u = find(u);
        v = find(v);
        if(u == v) return false;
        fa[u] = v;
        return true;
    }
};

Graph build_graph(MatrixCSC L) {
    Graph g;
    int n = L.n;
    g.n = n;

    for(int i = 0; i < n; i++) {
        for(int k = L.c[i]; k < L.c[i+1]; k++) {
            int j = L.r[k];
            if(i == j) continue;
            g.adde(i, j);
        }
    }
    return g;
}

CGraph compress_graph(Graph &g) {
    CGraph cg;
    int n = g.n;
    cg.n = n;
    cg.head = new int[n+1];
    int num_edges = 0;
    for(const auto& pr : g.e)
        num_edges += pr.second.size();
    cg.e = new int[num_edges];
    int cur_index = 0;
    for(int u = 0; u < n; u++) {
        cg.head[u] = cur_index;
        if(g.e.count(u) == 0) continue;
        const auto &e = g.e[u];
        for(int v : e) {
            cg.e[cur_index++] = v;
        }
    }
    cg.head[n] = cur_index;
    assert(cur_index == num_edges);
    return cg;
}

Partitions naive_partition(Graph &g) {
    vector<vector<vector<int>>> part;
    part.resize(1);
    part[0].resize(1);
    for(int i = 0; i < g.n; i++)
        part[0][0].push_back(i);
    Partitions p;
    p.init(part);
    return p;
}

vector<vector<int>> binpack(map<int,int> &cost, int worker) {
    int n = cost.size();
    vector<vector<int>> pack;
    vector<int> cost_sum;
    if(n <= worker) {
        for(auto &pr : cost) {
            pack.emplace_back(1, pr.first);
        }
    } else {
        pack.resize(worker);
        cost_sum.resize(worker);
        for(auto &pr : cost) {
            int r = pr.first;
            int c = pr.second;
            int imin = 0;
            for(int i = 1; i < worker; i++)
                if(cost_sum[i] < cost_sum[imin])
                    imin = i;
            pack[imin].push_back(r);
            cost_sum[imin] += c;
        }
    }
    return pack;
}

Partitions greedy_partition(Graph &g, int win, int workers) {
    int n = g.n;
    vector<int> indeg(n, 0);

    for(auto &pr : g.e) {
        auto &e = pr.second;
        for(int v : e) {
            indeg[v]++;
        }
    }

    vector<int> level(n, 0);
    queue<int> qu;
    for(int u = 0; u < n; u++)
        if(indeg[u] == 0) {
            qu.push(u);
            level[u] = 0;
        }

    vector<int> order;
    while(!qu.empty()) {
        int u = qu.front();
        order.push_back(u);
        qu.pop();
        for(int v : g.e[u]) {
            level[v] = std::max(level[v], level[u] + 1);
            indeg[v]--;
            if(indeg[v] == 0) {
                qu.push(v);
            }
        }
    }

//    vector<int> right(n, level[n - 1]);
//    for(int i = n - 2; i >= 0; i--) {
//        int u = order[i];
//    }
//
//    vector<int> pos(n, 0);
//    pos[n - 1] = level[n - 1];
//    for(int i = n - 2; i >= 0; i--) {
//        int u = order[i];
//        int pmin = level[u];
//        int pmax = n + 1;
//        for(int v : g.e[u])
//            pmax = std::min(pmax, pos[v]);
//        vector<int> au;
//        vector<int> apos;
//        vector<int> alev;
//        for(int v : g.e[u]) {
//            au.push_back(v);
//            apos.push_back(pos[v]);
//            alev.push_back(level[v]);
//        }
//
//        pos[u] = pmin + rand() % (pmax - pmin + 1);
//        if(pos[u] < level[u])
//            printf("NO\n");
//    }
//    level = pos;

    vector<set<int>> lsets(n, set<int>());
    int max_level = *std::max_element(level.begin(), level.end());
    for(int u = 0; u < n; u++)
        lsets[level[u]].insert(u);

    DisjointSet dset(n);
    vector<vector<vector<int>>> partition;
    for(int i = 0; i <= max_level; i += win) {
        int maxj = std::min(i + win - 1, max_level);
        for(int j = i; j <= maxj; j += 1) {
            for(int u : lsets[j]) {
                for(int v : g.e[u]) {
                    if(level[v] > maxj) continue;
                    dset.union_sets(u, v);
                }
            }
        }
        std::map<int, int> cost;
        for(int j = i; j <= maxj; j++) {
            for(int u : lsets[j]) {
                int r = dset.find(u);
                cost[r] += 1 + g.e[u].size();
            }
        }
        vector<vector<int>> pack = binpack(cost, workers);
        map<int,int> r2id;
        for(int j = 0; j < pack.size(); j++)
            for(int k = 0; k < pack[j].size(); k++)
                r2id[pack[j][k]] = j;
        vector<vector<int>> groups;
        groups.resize(pack.size());
        for(int j = i; j <= maxj; j++) {
            for(int u : lsets[j]) {
                int r = dset.find(u);
                groups[r2id[r]].push_back(u);
            }
        }
        for(int j = 0; j < groups.size(); j++)
            std::sort(groups[j].begin(), groups[j].end());
        partition.push_back(groups);
    }

    Partitions part;
    part.init(partition);
    return part;
}

void lsolve_dense_opt(MatrixCSC L, MatrixCSC b, CGraph cg, Partitions p) {
    assert(b.m == b.nnz && b.n == 1);

    int sum = 0;
    for(int i = 0; i < p.num_phase; i++)
        for(int j = 0; j < p.phase_workers[i]; j++)
            sum += p.lens[i][j];
    assert(L.n == sum);


//    int n = L.n;
//    int *r = L.r;
    int *c = L.c;
    dtype *bv = b.v;
    dtype *Lv = L.v;
    for(int i = 0; i < p.num_phase; i++) {
        int num_worker = p.phase_workers[i];
#pragma omp parallel num_threads(num_worker) default(none) shared(p, i, num_worker, c, bv, Lv, cg)
//        for(int q = 0; q < num_worker; q++)
        {
            int worker = omp_get_thread_num();
            int len = p.lens[i][worker];
            int* u_list = p.nodes[i][worker];
            for(int k = 0; k < len; k++) {
                int u = u_list[k];
                int col_start = c[u];
                int j_start = cg.head[u];
                bv[u] /= Lv[col_start];
                for(int j = cg.head[u]; j < cg.head[u+1]; j++) {
                    int v = cg.e[j];
                    dtype tmp = bv[u] * Lv[col_start + j - j_start + 1];
#pragma omp atomic
                    bv[v] -= tmp;
                }
            }
        }
    }
}


