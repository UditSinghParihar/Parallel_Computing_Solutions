#include <iostream>
#include <vector>
#include "dvec.h"

using namespace std;

#define BlockMax 65535
#define ThreadMax 1024

__global__ void add(float *x, float *y, int n){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;

	// if(index==0)
	// 	printf("Blocks: %d\tThreads: %d\n--\n", gridDim.x, blockDim.x);
	
	for(int i=index; i<n; i+=stride)
		y[i] += x[i]; 
}

void print_vector(const vector<float>& vec){
	for(auto element : vec){;
		fprintf(stdout, "%9.4f\n", element);
	}
	cout << "\n---\n";
}

int main(int argc, char const *argv[]){
	const int N = 10;
	vector<float> x(N);
	vector<float> y(N);

	for(int i=0; i<x.size(); ++i){
		x[i] = i;
		y[i] = 2.0*i;
	}
	print_vector(x);
	print_vector(y);

	// float *d_x, *d_y;
	// size_t sz = N*sizeof(float);
	// cudaMalloc(&d_x, sz);
	// cudaMalloc(&d_y, sz);
	// cudaMemcpy(d_x, x.data(), sz, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_y, y.data(), sz, cudaMemcpyHostToDevice);

	// fprintf(stdout, "Kernel started.\n");
	// add<<<(N+ThreadMax-1)/ThreadMax, ThreadMax>>>(d_x, d_y, N);
	// cudaDeviceSynchronize();
	// fprintf(stdout, "Kernel ended. \n");

	// cudaMemcpy(y.data(), d_y, sz, cudaMemcpyDeviceToHost);
	// // print_vector(y);

	// cudaFree(d_x);
	// cudaFree(d_y);

	dvec<float> d_x(N);
	dvec<float> d_y(N);

	d_x.set(&x[0]);
	d_y.set(&y[0]);

	add<<<(N+ThreadMax-1)/ThreadMax, ThreadMax>>>(d_x.data(), d_y.data(), N);
	cudaDeviceSynchronize();
	d_y.get(&y[0]);

	print_vector(y);	


	return 0;
}