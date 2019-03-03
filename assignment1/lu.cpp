#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>

using namespace std;

void fill_matrix(vector<vector<float>>& mat, int range){
	const int rows = mat.size(), cols = mat[0].size();
	for(int i=0; i<rows; ++i){
		for(int j=0; j<cols; ++j){
			mat[i][j] = (rand() % range) + 1; 
		}
	}
}

void fill_hessenberg(vector<vector<float>>& mat, int range){
	const int rows = mat.size(), cols = mat[0].size();
	for(int i=0; i<rows; ++i){
		for(int j=0; j<cols; ++j){
			if(i-j>1)
				mat[i][j] = 0;
			else
				mat[i][j] = (rand() % range) + 1;
		}
	}
}

void print_matrix(const vector<vector<float>>& mat){
	for(auto v : mat){
		for(auto element : v){
			fprintf(stdout, "%9.2f", element);
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
	for(auto element : vec){
		cout << element << endl;
	}
	cout << "\n---\n";
}

void my_lu(vector<vector<float>>& A){
	const int rows = A.size(), cols = A[0].size();
	for(int k=0; k<cols-1; ++k){
		for(int p=k+1; p<rows; ++p){
			A[p][k] = A[p][k]/A[k][k];
		}
		for(int i=k+1; i<rows; ++i){
			for(int j=k+1; j<cols; ++j){
				A[i][j] -= A[i][k]*A[k][j];
			}
		}
	}
}

void my_lu_hessenberg(vector<vector<float>>& A){
	const int rows = A.size(), cols = A[0].size();
	for(int k=0; k<cols-1; ++k){
		A[k+1][k] = A[k+1][k]/A[k][k];
		for(int j=k+1; j<cols; ++j){
			A[k+1][j] -= A[k+1][k]*A[k][j];
		}
	}
}

void get_lu(const vector<vector<float>>& A, vector<vector<float>>& L,
		vector<vector<float>>& U){
	const int rows = A.size();
	for(int i=0; i<rows; ++i){
		for(int j=0; j<rows; ++j){
			if(i == j){
				L[i][j] = 1;
				U[i][j] = A[i][j];
			}
			else if(i > j){
				L[i][j] = A[i][j];
			}
			else if(j > i){
				U[i][j] = A[i][j];	
			}
		}
	}
}

void forward(const vector<vector<float>>& L, vector<float>& t, const vector<float>& b){
	const int rows = L.size(), cols = L[0].size();
	
	t[0] = b[0]/L[0][0];
	float sum=0;
	for(int i=1; i<rows; ++i){
		sum = b[i];
		for(int j=0; j<=i-1; ++j){
			sum -= L[i][j]*t[j];
		}
		t[i] = sum/L[i][i];
	}
}

void backward(const vector<vector<float>>& U, vector<float>& x, const vector<float>& t){
	const int rows = U.size(), cols = U[0].size();

	x[rows-1] = t[rows-1]/U[rows-1][rows-1];
	float sum = 0;
	for(int i=rows-2; i>=0; --i){
		sum = t[i];
		for(int j=rows-1; j>=i+1; --j){
			sum -= U[i][j]*x[j];
		}
		x[i] = sum/U[i][i];
	}
	cout << "\n\n";
}

float get_error(const vector<vector<float>>& A, const vector<float>& x,
				const vector<float>& b){
	const int rows = A.size(), cols = A[0].size();
	vector<float> res(rows);
	for(int i=0; i<rows; ++i){
		for(int j=0; j<cols; ++j){
			res[i] += A[i][j]*x[j];
		}
		res[i] -= b[i];
	}
	return sqrt(inner_product(res.begin(), res.end(), res.begin(), 0));
}

void solve_lu(void){
	const int rows_a=3, cols_a=3, range=10;
	vector<vector<float>> A(rows_a, vector<float>(cols_a));
	fill_matrix(A, range);
	vector<vector<float>> A_copy(A);
	print_matrix(A);

	my_lu(A);
	vector<vector<float>> L(rows_a, vector<float>(cols_a)), U(rows_a, vector<float>(cols_a));
	get_lu(A, L, U);

	vector<float> x(rows_a), b(rows_a), t(rows_a);
	fill_vector(b, range);
	print_vector(b);

	forward(L, t, b);
	backward(U, x, t);
	print_vector(x);

	cout << "Error is: " << get_error(A_copy, x, b) << "\n\n";
}

void solve_hessenberg(void){
	const int rows_h=3, cols_h=3, range_h=10;
	vector<vector<float>> H(rows_h, vector<float>(cols_h));
	fill_hessenberg(H, range_h);
	print_matrix(H);

	my_lu_hessenberg(H);
	print_matrix(H);
}

int main(int argc, char const *argv[]){
	// Question 2.a and 2.b : LU, Foreword, Backward, solve
	solve_lu();

	// Question 2.c, 3.c : Hessenberg
	solve_hessenberg();

	return 0;
}