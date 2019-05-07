#include <iostream>
#include "linear_algebra.h"
#include "dvec.h"

using namespace std;
using namespace la;


int main(int argc, char const *argv[]){
	const int rows_a=3, cols_a=3, range=10;
	Matrix A(rows_a, Vector(cols_a));
	Matrix B(rows_a, Vector(cols_a));
	Matrix C(rows_a, Vector(cols_a));
	fill_matrix(A, range);
	fill_matrix(B, range);
	print_matrix(A);
	print_matrix(B);

	// Matrix A_red(rows_a, Vector(cols_a));
	// const int k_col = 4;
	// pca(A, A_red, k_col);
	// print_matrix(A_red);

	// compress(argv[1]);

	dvec<float> d_A(rows_a*cols_a);
	dvec<float> d_B(rows_a*cols_a);
	dvec<float> d_C(rows_a*cols_a);
	d_A.set(&A[0][0]);
	d_B.set(&B[0][0]);
	matrixMul(d_A.data(), d_B.data(), d_C.data(), rows_a);
	cudaDeviceSynchronize();
	d_C.get(&C[0][0]);

	print_matrix(C);

	return 0;
}