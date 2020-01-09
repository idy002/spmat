cmake_minimum_required (VERSION 2.8.11)

project(spmat)

add_executable(main main.cpp csc.cpp lsolve_dense.cpp lsolve_sparse.cpp spmv.cpp spmv.h lsolve_dense_opt.cpp)
target_compile_options(main PUBLIC -fopenmp -O2 -Wall)
#target_compile_options(main PUBLIC -fopenmp -Wall)
target_link_libraries(main gomp)



