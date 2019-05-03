#include <iostream>
#include "linear_algebra.h"

using namespace std;
using namespace la;

int main(int argc, char const *argv[]){
	const int rows=5, range=10;
	// vector<float> x(rows);
	// fill_vector(x, range);
	// print_vector(x);

	// Vector x{-2.0913,-4.1066};
	// print_vector(x);
	// vector<vector<float>> P(2, vector<float>(2));
	// householder(x, P);
	// print_matrix(P);
	// check_householder(P, x);
	
	const int rows_a=3, cols_a=3;
	Matrix A(rows_a, Vector(cols_a));
	fill_matrix(A, range);
	print_matrix(A);

	Matrix R(rows_a, Vector(cols_a));
	Matrix Q(rows_a, Vector(cols_a));
	qr(A, Q, R);
	print_matrix(Q);
	print_matrix(R);

	// vector<vector<float>> B(rows_a, vector<float>(cols_a));
	// vector<vector<float>> C(rows_a, vector<float>(cols_a));
	// fill_matrix(B, range);
	// print_matrix(B);

	// matrix_mul(A, B, C, 1,1,1,1);
	// print_matrix(C);

	// vector<float> x(rows);
	// vector<vector<float>> D(rows, vector<float>(rows));
	// fill_vector(x, range);
	// print_vector(x);		
	
	// outer_product(x, D);
	// print_matrix(D);

	return 0;
}