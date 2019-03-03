#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>

using namespace std;

void fill_matrix(vector<vector<int>>& mat, int range){
	const int rows = mat.size(), cols = mat[0].size();
	for(int i=0; i<rows; ++i){
		for(int j=0; j<cols; ++j){
			mat[i][j] = (rand() % range) + 1; 
		}
	}
}

void fill_sparse_matrix(vector<vector<int>>& mat, int range){
	const int rows = mat.size(), cols = mat[0].size();
	int count=0;
	for(int i=0; i<rows; ++i){
		for(int j=0; j<cols; ++j){
			count = (i*cols)+j;
			if((count % (rows/2)) == 0){
				mat[i][j] = (rand() % range) + 1;
			}
			else{
				mat[i][j] = 0;		
			}
		}
	}
}

void get_coo_matrix(const vector<vector<int>>& mat, vector<int>& row,
					vector<int>& col, vector<int>& value){
	const int rows = mat.size(), cols = mat[0].size();
	for(int i=0; i<rows; ++i){
		for(int j=0; j<cols; ++j){
			if(mat[i][j] != 0){
				value.push_back(mat[i][j]);
				row.push_back(i);
				col.push_back(j);						
			} 
		}
	}		
}

void coo_matrix_vector(const vector<int>& row, const vector<int>& col, const 
				vector<int>& value, const vector<int>& list, vector<int>& result){
	for(int i=0; i<value.size(); ++i){
		result[row[i]] += value[i]*list[col[i]];
	}
}

void fill_vector(vector<int>& col, int range){
	for(int i=0; i<col.size(); ++i){
		col[i] = (rand() % range) + 1;
	}
}

void print_matrix(const vector<vector<int>>& mat){
	for(auto v : mat){
		for(auto element : v){
			fprintf(stdout, "%4d", element);
		}
		cout << endl;
	}
	cout << "\n---\n";
}

void print_vector(const vector<int>& vec){
	for(auto element : vec){
		cout << element << endl;
	}
	cout << "\n---\n";
}

void matrix_multiply(const vector<vector<int>>& A, const vector<vector<int>>& B, 
					vector<vector<int>>& C){
	const int rows_a = A.size(), cols_a = A[0].size();
	const int rows_b = B.size(), cols_b = B[0].size();
	for(int i=0; i<rows_a; ++i){
		for(int j=0; j<cols_b; ++j){
			for(int k=0; k<cols_a; ++k){
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

void fill_band_matrix(vector<vector<int>>& mat, int p, int q, int range){
	const int rows = mat.size(), cols = mat[0].size();
	for(int i=0; i<rows; ++i){
		for(int j=0; j<cols; ++j){
			if((i-j <= p && i-j >= 0) || (i-j >= -q && i-j <= -1)){
				mat[i][j] = (rand() % range) + 1; 
			}
		}
	}
}

void get_band_matrix(const vector<vector<int>>& mat, vector<vector<int>>& mat_band,
						int p, int q){
	const int rows = mat.size(), cols = mat[0].size();
	for(int i=0; i<rows; ++i){
		for(int j=0; j<cols; ++j){
			if((i-j <= p && i-j >= 0) || (i-j >= -q && i-j <= -1)){
				mat_band[i][j-i+p] = mat[i][j];
			}
		}
	}	
}

void band_matrix_vector(const vector<vector<int>>& A, const vector<int>& col,
						vector<int>& result, int p, int q){
	const int rows = result.size();
	for(int i=0; i<rows; ++i){
		int j_start = max(0, i-p), j_stop = min(rows, i+q);
		for(int j=j_start; j<=j_stop && j<rows; ++j){
			result[i] += A[i][j] * col[j];
		}
	}
}

void solve_dense(void){
	const int rows_a=4, cols_a=5, rows_b=5, cols_b=3;
	vector<vector<int>> A(rows_a, vector<int>(cols_a));
	vector<vector<int>> B(rows_b, vector<int>(cols_b));

	const int range = 20;
	fill_matrix(A, range);
	fill_matrix(B, range);

	vector<vector<int>> C(rows_a, vector<int>(cols_b));
	matrix_multiply(A, B, C);
}

void solve_banded(void){
	const int rows_d = 6, cols_d = 6, p = 1, q = 2, range_d = 20;
	vector<vector<int>> D(rows_d, vector<int>(cols_d));
	fill_band_matrix(D, p, q, range_d);
	print_matrix(D);

	vector<vector<int>> D_band(rows_d, vector<int>(p+q+1));
	get_band_matrix(D, D_band, p, q);

	vector<int> col_d(cols_d);
	fill_vector(col_d, 20);
	print_vector(col_d);

	vector<int> result_d(cols_d);
	band_matrix_vector(D, col_d, result_d, p, q);
	print_vector(result_d);
}

void solve_coo(void){
	const int rows_e = 7, cols_e = 7, range_e = 20;
	vector<vector<int>> E(rows_e, vector<int>(cols_e));
	fill_sparse_matrix(E, range_e);
	print_matrix(E);

	vector<int> row_index, col_index, value;
	get_coo_matrix(E, row_index, col_index, value);

	vector<int> col_e(cols_e);
	fill_vector(col_e, 20);
	print_vector(col_e);

	vector<int> result_e(cols_e);
	coo_matrix_vector(row_index, col_index, value, col_e, result_e);
	print_vector(result_e);
}

int main(int argc, char const *argv[]){
	// Question 1.1 : Dense
	solve_dense();

	// Question 1.2 : Banded
	solve_banded();

	// Question 1.3.a : COO
	solve_coo();
	
	return 0;
}