#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>

using namespace std;

typedef vector<float> Vector;
typedef vector<Vector> Matrix;

void fill_matrix(vector<vector<float>>& mat, int range){
	const int rows = mat.size(), cols = mat[0].size();
	for(int i=0; i<rows; ++i){
		for(int j=0; j<cols; ++j){
			mat[i][j] = (rand() % range) + 1; 
		}
	}
}

void print_matrix(const vector<vector<float>>& mat){
	for(auto v : mat){
		for(auto element : v){
			fprintf(stdout, "%9.4f", element);
		}
		cout << endl;
	}
	cout << "\n---\n";
}

void fill_vector(vector<float>& col, int range){
	for(int i=0; i<col.size(); ++i){
		col[i] = (rand() % range) + 1;
	}
}

void print_vector(const vector<float>& vec){
	for(auto element : vec){;
		fprintf(stdout, "%9.4f\n", element);
	}
	cout << "\n---\n";
}

void check_householder(const vector<vector<float>>& P, const vector<float>& x){
	const int rows = P.size();
	vector<float> res(rows);

	for(int i=0; i<rows; ++i){
		for(int j=0; j<rows; ++j){
			res[i] += P[i][j]*x[j];
		}
	}

	cout << "P*x = \n";
	print_vector(res);
	
	float x_norm=0.0, sum=0.0;
	for(int i=0; i<x.size(); ++i)
		sum += x[i]*x[i];
	x_norm = sqrt(sum);

	cout << "x_norm = " << x_norm << endl;
}

void householder(vector<float> u, vector<vector<float>>& P){	
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

Matrix matrix_mul(const vector<vector<float>>& A, const vector<vector<float>>& B){
	const int rows_a = A.size(), cols_a = A[0].size();
	const int rows_b = B.size(), cols_b = B[0].size();
	Matrix C(rows_a, Vector(cols_b));

	for(int i=0; i<rows_a; ++i){
		for(int k=0; k<cols_b; ++k){
			for(int j=0; j<cols_a; ++j){
				C[i][k] += A[i][j]*B[j][k];
			}
		}
	}

	return C;
}

void matrix_mul(const vector<vector<float>>& A, const vector<vector<float>>& B, vector<vector<float>>& C,
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

void qr(const vector<vector<float>>& A, vector<vector<float>>& Q, vector<vector<float>>& R){
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
		vector<float> u(size);
		vector<vector<float>> H(size, vector<float>(size));
		
		for(int j=i, jj=0; j<rows; ++j, ++jj)
			u[jj] = R[j][i];
	
		householder(u, H);
	
		vector<vector<float>> Q_sub(rows, vector<float>(cols));
		vector<vector<float>> R_sub(rows, vector<float>(cols));

		matrix_mul(H, R, R_sub, 0, 0, i, i);
		matrix_mul(Q, H, Q_sub, 0, i, 0, 0);
		
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
	vector<vector<float>> A(rows_a, vector<float>(cols_a));
	fill_matrix(A, range);
	print_matrix(A);

	vector<vector<float>> R(rows_a, vector<float>(cols_a));
	vector<vector<float>> Q(rows_a, vector<float>(cols_a));
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