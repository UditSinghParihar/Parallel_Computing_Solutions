# TODO #

1. Add assertion
2. Make generalize for rectangle matrix(Currently square matrix).
3. Profiling of critical sections.
4. Incorporate OpenMP in CPU code.

# Info #

1. Max Blocks: 65535 blocks/grid
2. Max Threads: 1024 threads/block
3. Compile - 
	1. `C++:-`  `g++ pca.cpp linear_algebra.cpp -std=c++11 pkg-config --cflags --libs opencv`
	2. `Cuda:-` `nvcc la.cu -std=c++11`
