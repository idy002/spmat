# spmat
A simple sparse triangular solver.

### Usage
- download the matrices in experiments: 
[L](https://sparse.tamu.edu/Schenk_AFE/af_0_k101), 
[dense b](https://drive.google.com/file/d/1L8eDJ0ADXgTJZysNLshiPdW3lFAs33B5/view?usp=sharing) and 
[sparse b](https://drive.google.com/file/d/1zZeL2V8J_bdp5dZBJgjbwe1ezGl71p3O/view?usp=sharing).

Place them in `spmat/data/af_0_k101` sub-directory

- enter repo directory
```shell script
cd spmat
```

- build the project
```shell script
mkdir build
cd build
cmake ..
make 
cd ..
```

- run solver
```shell script
./build/main
```

- Below is the result in my laptop ():
```text
reading matrices... done
dense rhs, naive, elapsed time: 10.973422 ms
MAE: 0.000001077699608

dense rhs, optimized, elapsed time: 79.994741 ms
MAE: 0.000001112970835

enumerate different win:
1: 103.951583
2: 88.608399
3: 75.444612
4: 75.229748
5: 75.218779
6: 74.898024
7: 75.754060
8: 75.468960
9: 74.679102
10: 75.442105
11: 74.897580
12: 96.612315
13: 78.456274
14: 75.704154
15: 73.961080
16: 75.360023
17: 76.053466
18: 76.088654
19: 75.411162
20: 77.105087
```
The result shows that the naive implementation takes 11.0 ms while the parallel optimized implementation takes 80.0 ms.
I think the slowdown comes from the overhead of threads creation/deletion in OpemMP.

In the optimized implementation, I construct the dependency DAG of the triangular matrix. 
Then use a heuristic to partition the DAG into multiple layers. 
Each layer has at most #worker components that would be executed concurrently in parallel threads created by OpenMP.

The heuristic is simple: 
- Set the level of each node in DAG as the longest distance to a zero-indegree node in the DAG.
- Given parameter `win`, partition the DAG into multiple layers. The first layer contains the nodes with level 0 to win-1, the second layer contains nodes with level win to 2*win-1, and etc.
- For each layer, use binpack to partition connected components into at most #workers larger components.
- During execution, the solver would process layer by layer. For each layer, the components in the layer would be processed parallelly.
(The heuristic is a simplified version of ParSy)

### Verification
To verify the correctness of the solver of `Lx=b`. I do the matrix-vector multiplication of `L` and the result of solver `x` to get vector `c`: `Lx=c`. 
Then compare the original right hand side vector `b` and `c`. The mean absolute error (MAE) is reported in the output of program.

### Sources code

```text
spmat
    data
        af_0_k101   # matrices used in experiment
            af_0_k101.mtx
            b_dense_af_0_k101.mtx
            b_sparse_af_0_k101.mtx
        sample      # matrices used in debugging
            L.mtx
            b_dense.mtx
            b_sparse.mtx
    csc.h       # process Compressed Sparse Column format
    csc.cpp 
    spmv.h      # naive implementation of sparse matrix-vector multiplication
    spmv.cpp
    lsolve.h    # declare the triangular matrix solver system of naive, optimized implementation
    lsolve_dense.cpp        # naive implementation of triangular matrix solver (dense rhs vector)
    lsolve_dense_opt.cpp    # optimized implementation of triangular matrix solver (dense rhs vector)
    lsolve_sparse.cpp       # naive implementation of triangular matrix solver (sparse rhs vector)
    main.cpp    # main process
    README.md
```

### Question & Problems
The optimized implementation gets slower. One reason might because the matrix is too small: the naive implementation only takes 11 ms. 
The other reason might come from the overhead of thread process in OpenMP. 

I do not know whether I have used OpenMP correctly. I would appreciate it if you could help me to verify the optimized implementation.

