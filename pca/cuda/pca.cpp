#include <iostream>
#include "linear_algebra.h"

using namespace std;
using namespace la;


int main(int argc, char const *argv[]){
	const int rows_a=20, cols_a=8, range=10;
	Matrix A(rows_a, Vector(cols_a));
	fill_matrix(A, range);
	print_matrix(A);

	Matrix A_red(rows_a, Vector(cols_a));
	const int k_col = 4;
	pca(A, A_red, k_col);
	print_matrix(A_red);

	// compress(argv[1]);

	return 0;
}