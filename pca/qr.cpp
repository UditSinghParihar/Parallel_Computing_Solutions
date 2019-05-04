#include <iostream>
#include "linear_algebra.h"

using namespace std;
using namespace la;

void mean_col(const Matrix& A, Matrix& m){
	const int rows = A.size(), cols = A[0].size();

	for(int i=0; i<cols; ++i){
		float sum = 0.0;

		for(int j=0; j<rows; ++j)
			sum += A[j][i];
		sum = sum/rows;

		for(int k=0; k<rows; ++k)
			m[k][i] = sum;
	}
}

void mat_sub(const Matrix& A, const Matrix& B, Matrix& C){
	const int rows = A.size(), cols = A[0].size();

	for(int i=0; i<rows; ++i)
		for(int j=0; j<cols; ++j)
			C[i][j] = A[i][j] - B[i][j];
}

void mat_add(const Matrix& A, const Matrix& B, Matrix& C){
	const int rows = A.size(), cols = A[0].size();

	for(int i=0; i<rows; ++i)
		for(int j=0; j<cols; ++j)
			C[i][j] = A[i][j] + B[i][j];
}

void pca(const Matrix& A, Matrix& A_red, const int k_col){
	const int rows = A.size(), cols = A[0].size();

	Matrix m(rows, Vector(cols));
	mean_col(A, m);
	mat_sub(A, m, A_red);

	Matrix cov(rows, Vector(cols));
	mat_mul(trans(A_red), A_red, cov, 0, 0, 0, 0);
	
	Matrix V(rows, Vector(cols));
	Matrix D(rows, Vector(cols));
	eig(cov, V, D);

	Matrix v_k(rows, Vector(k_col));
	for(int i=0; i<k_col; ++i)
		for(int j=0; j<rows; ++j)
			v_k[j][i] = V[j][i];


	mat_add(mat_mul(A_red, mat_mul(v_k, trans(v_k))), m, A_red);
}


int main(int argc, char const *argv[]){
	const int rows_a=5, cols_a=5, range=10;
	Matrix A(rows_a, Vector(cols_a));
	fill_matrix(A, range);
	print_matrix(A);

	Matrix A_red(rows_a, Vector(cols_a));
	const int k_col = 2;
	pca(A, A_red, k_col);
	print_matrix(A_red);

	return 0;
}