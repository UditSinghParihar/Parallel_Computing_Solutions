#include <iostream>
#include <limits>
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include "linear_algebra.h"

using namespace std;
using namespace la;

void sort_index(Vector& v, Vector& idx){
	iota(idx.begin(), idx.end(), 0);
	sort(idx.begin(), idx.end(), [&v](int i1, int i2){return v[i1] > v[i2];});
	sort(v.begin(), v.end(), std::greater<float>());
}

void sort_eigens(Matrix& V, Matrix& D){
	const int rows = D.size(), cols = D[0].size();
	
	Vector d(cols);
	Vector idx(cols);
	for(int i=0; i<cols; ++i)
		d[i] = D[i][i];

	sort_index(d, idx);
	
	for(int i=0; i<cols; ++i)
		D[i][i] = d[i];

	Matrix V_tmp(V);
	for(int i=0; i<cols; ++i)
		for(int j=0; j<rows; ++j)
			V[j][i] = V_tmp[j][idx[i]];
		
}

void eig(const Matrix& A, Matrix& V, Matrix& D){
	const int rows = A.size(), cols = A[0].size();

	D = A;
	for(int i=0; i<cols; ++i)
		V[i][i] = 1;

	const float eps = 1e-15;
	bool is_converged = false;
	float sum_prev = numeric_limits<float>::lowest();
	float change = 0.0;
	int steps = 0;

	while(!is_converged){
		Matrix Q(rows, Vector(cols));
		Matrix R(rows, Vector(cols));
		qr(D, Q, R);

		D = mat_mul(R, Q);
		V = mat_mul(V, Q);
		
		float sum = trace(D);
		change = sum - sum_prev;
		if(abs(change) < eps)
			is_converged = true;
		
		sum_prev = sum;
		++steps;
	}

	sort_eigens(V, D);

	cout << "Steps: " << steps << "\n--\n";
}

int main(int argc, char const *argv[]){
	const int rows_a=3, cols_a=3, range=10;
	Matrix A(rows_a, Vector(cols_a));
	fill_matrix(A, range);
	print_matrix(A);

	Matrix V(rows_a, Vector(cols_a));
	Matrix D(rows_a, Vector(cols_a));
	eig(mat_mul(trans(A), A), V, D);
	print_matrix(V);
	print_matrix(D);

	return 0;
}