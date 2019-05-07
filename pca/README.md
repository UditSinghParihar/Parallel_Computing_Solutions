## Description ##

1. Implementation of PCA from scratch, along with implementation of QR and Eigen Decomposition for image compression.
2. Algorithms are implemented in both `Matlab` and `C++`/`CUDA`to bencmark the parallelism.

2. **Directory Structure:**
	1. *matlab:*
		1. `householder.m` : Calculates householder matrix of a vector.
		2. `my_qr.m` : Calculates QR Decomposition of matrix.
		3. `my_eig.m` : Calculates Eigenvectors and Eigenvalues of a matrix.
		4. `my_pca.m` : Implements PCA over a image to give low dimensional image.

	2. *cpp:*
		1. `linear_algebra.h` : Header file containing all the matrix operations required to implement pca.
		2. `pca.cpp` : Main function to convert image data to suitable format and calls pca function. 

	3. *cuda:*
		1. `dvec.h` : `vector class container` used by device(GPU).
		2.  `linear_algebra.cu` : Parallel matrix multiplication and other sequential matrix operations.

3. [Project Report](https://github.com/UditSinghParihar/Parallel_Computing_Solutions/blob/master/pca/results/pca_slides.pdf)