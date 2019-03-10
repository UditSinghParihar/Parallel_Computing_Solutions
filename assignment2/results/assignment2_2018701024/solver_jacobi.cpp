#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

using namespace std;
extern double eps;
extern double** New_Matrix(int m, int n);
extern void Delete_Matrix(double** matrix);

int solver(double** a, int n){
	int k = 0;
	double diff=0;
	double** b = New_Matrix(n,n);
	if(b == NULL){
		cerr << "Jacobi: Can’t allocate matrix\n";
		exit(1);
	}
	
	do{
		diff = 0;
		#pragma omp parallel for schedule(static, 100) collapse(2) reduction(max : diff) 
		for(int i=1; i<n-1; i++){
			for(int j=1; j<n-1; j++){
				b[i][j] = 0.25 * (a[i][j-1] + a[i-1][j]
				+ a[i+1][j] + a[i][j+1]);
				double h = fabs(a[i][j] - b[i][j]);
				if(h > diff)
					diff = h;
			}
		}

		#pragma omp parallel for schedule(static, 100) collapse(2)
		for(int i=1; i<n-1; i++){
			for(int j=1; j<n-1; j++){
				a[i][j] = b[i][j];
			}
		}

		k++;
	}while(diff > eps);
	

	Delete_Matrix(b);
	
	return k;
}