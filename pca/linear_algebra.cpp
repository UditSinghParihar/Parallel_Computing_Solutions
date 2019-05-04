#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include "linear_algebra.h"

using namespace std;

void la::fill_matrix(la::Matrix& mat, int range){
	const int rows = mat.size(), cols = mat[0].size();
	for(int i=0; i<rows; ++i){
		for(int j=0; j<cols; ++j){
			mat[i][j] = (rand() % range) + 1; 
		}
	}
}

void la::print_matrix(const la::Matrix& mat){
	for(auto v : mat){
		for(auto element : v){
			fprintf(stdout, "%9.4f", element);
		}
		cout << endl;
	}
	cout << "\n---\n";
}

void la::fill_vector(la::Vector& col, int range){
	for(int i=0; i<col.size(); ++i){
		col[i] = (rand() % range) + 1;
	}
}

void la::print_vector(const la::Vector& vec){
	for(auto element : vec){;
		fprintf(stdout, "%9.4f\n", element);
	}
	cout << "\n---\n";
}

void la::check_householder(const la::Matrix& P, const la::Vector& x){
	const int rows = P.size();
	la::Vector res(rows);

	for(int i=0; i<rows; ++i){
		for(int j=0; j<rows; ++j){
			res[i] += P[i][j]*x[j];
		}
	}

	cout << "P*x = \n";
	la::print_vector(res);
	
	float x_norm=0.0, sum=0.0;
	for(int i=0; i<x.size(); ++i)
		sum += x[i]*x[i];
	x_norm = sqrt(sum);

	cout << "x_norm = " << x_norm << endl;
}

void la::householder(la::Vector u, la::Matrix& P){	
	float u_norm=0.0, sum=0.0;
	for(int i=0; i<u.size(); ++i)
		sum += u[i]*u[i];
	u_norm = sqrt(sum);

	if(u[0] < 0)
		u_norm = -u_norm;

	u[0] += u_norm;
	float tau = u[0]/u_norm;
	
	float first_ele = u[0];
	for(auto& ele : u)
		ele = ele/first_ele;

	const int rows = P.size();
	for(int i=0; i<rows; ++i){
		for(int j=0; j<rows; ++j){
			if(i==j)
				P[i][j] = 1 - tau*u[i]*u[j];	
			else
				P[i][j] = -tau*u[i]*u[j]; 
		}
	}
}

la::Matrix la::mat_mul(const la::Matrix& A, const la::Matrix& B){
	const int rows_a = A.size(), cols_a = A[0].size();
	const int rows_b = B.size(), cols_b = B[0].size();
	la::Matrix C(rows_a, la::Vector(cols_b));

	for(int i=0; i<rows_a; ++i){
		for(int k=0; k<cols_b; ++k){
			for(int j=0; j<cols_a; ++j){
				C[i][k] += A[i][j]*B[j][k];
			}
		}
	}

	return C;
}

void la::mat_mul(const la::Matrix& A, const la::Matrix& B, la::Matrix& C,
	int a_row_begin=0, int a_col_begin=0, int b_row_begin=0, int b_col_begin=0){
	
	const int rows_a = A.size(), cols_a = A[0].size();
	const int rows_b = B.size(), cols_b = B[0].size();

	for(int i=a_row_begin, ii=0; i<rows_a; ++i, ++ii){
		for(int k=b_col_begin, kk=0; k<cols_b; ++k, ++kk){
			for(int j=a_col_begin, jj=b_row_begin; j<cols_a; ++j, ++jj){
				C[ii][kk] += A[i][j]*B[jj][k];
			}
		}
	}
}

void la::qr(const la::Matrix& A, la::Matrix& Q, la::Matrix& R){
	const int rows = A.size(), cols = A[0].size();

	for(int i=0; i<rows; ++i){
		for(int j=0; j<cols; ++j){
			if(i==j)
				Q[i][j] = 1;
			R[i][j] = A[i][j];
		}
	}
	for(int i=0; i<cols; ++i){
		int size = cols-i;
		la::Vector u(size);
		la::Matrix H(size, la::Vector(size));
		
		for(int j=i, jj=0; j<rows; ++j, ++jj)
			u[jj] = R[j][i];
	
		householder(u, H);
	
		la::Matrix Q_sub(rows, la::Vector(cols));
		la::Matrix R_sub(rows, la::Vector(cols));

		mat_mul(H, R, R_sub, 0, 0, i, i);
		mat_mul(Q, H, Q_sub, 0, i, 0, 0);
		
		for(int k=0; k<size; ++k){
			for(int l=0; l<size; ++l){
				R[i+k][i+l] = R_sub[k][l];
			}
		}
		
		for(int k=0; k<rows; ++k){
			for(int l=0; l<size; ++l){
				Q[k][i+l] = Q_sub[k][l]; 
			}
		}
	
	}
}

float la::trace(const la::Matrix& A){
	const int cols = A[0].size();

	float sum = 0.0;
	for(int i=0; i<cols; ++i)
		sum += A[i][i];

	return sum;
}

la::Matrix la::trans(const la::Matrix& mat){
	const int rows = mat.size(), cols = mat[0].size();
	la::Matrix tr(cols, la::Vector(rows));

	for(int i=0; i<rows; ++i)
		for(int j=0; j<cols; ++j)
			tr[j][i] = mat[i][j]; 

	return tr;
}

void la::sort_index(la::Vector& v, la::Vector& idx){
	iota(idx.begin(), idx.end(), 0);
	sort(idx.begin(), idx.end(), [&v](int i1, int i2){return v[i1] > v[i2];});
	sort(v.begin(), v.end(), std::greater<float>());
}

void la::sort_eigens(la::Matrix& V, la::Matrix& D){
	const int rows = D.size(), cols = D[0].size();
	
	la::Vector d(cols);
	la::Vector idx(cols);
	for(int i=0; i<cols; ++i)
		d[i] = D[i][i];

	sort_index(d, idx);
	
	for(int i=0; i<cols; ++i)
		D[i][i] = d[i];

	la::Matrix V_tmp(V);
	for(int i=0; i<cols; ++i)
		for(int j=0; j<rows; ++j)
			V[j][i] = V_tmp[j][idx[i]];
		
}

void la::eig(const la::Matrix& A, la::Matrix& V, la::Matrix& D){
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
		la::Matrix Q(rows, la::Vector(cols));
		la::Matrix R(rows, la::Vector(cols));
		la::qr(D, Q, R);

		D = la::mat_mul(R, Q);
		V = la::mat_mul(V, Q);
		
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