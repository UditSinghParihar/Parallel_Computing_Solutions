#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>

using namespace std;

void print_matrix(const vector<vector<float>>& mat){
	for(auto v : mat){
		for(auto element : v){
			fprintf(stdout, "%9.3f", element);
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

void fill_v(const vector<float>& x, vector<float>& v){
	const int rows = x.size();	
	const float x_norm = sqrt(inner_product(x.begin(), x.end(), x.begin(), 0));
	
	v[0] = 1;
	for(int i=1; i<rows; ++i){
		v[i] = x[i]/(x[0]-x_norm);
	}
}

void get_householder(vector<vector<float>>& P, const vector<float>& v){
	const int rows = P.size();
	const float v_norm = sqrt(inner_product(v.begin(), v.end(), v.begin(), 0.0));
	const float beta = 2/(v_norm*v_norm);
	
	for(int i=0; i<rows; ++i){
		for(int j=0; j<rows; ++j){
			if(i==j)
				P[i][j] = 1 - beta*v[i]*v[j];	
			else
				P[i][j] = -beta*v[i]*v[j]; 
		}
	}	
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
	float x_norm = sqrt(inner_product(x.begin(), x.end(), x.begin(), 0));
	cout << "x_norm = " << x_norm << endl;
}

int main(int argc, char const *argv[]){
	// 3.a : Householder

	const int rows=5, range=10;
	vector<float> x(rows);
	fill_vector(x, range);
	print_vector(x);

	vector<float> v(rows);
	fill_v(x, v);
	print_vector(v);

	vector<vector<float>> P(rows, vector<float>(rows));
	get_householder(P, v);
	print_matrix(P);
	check_householder(P, x);
	
	return 0;
}