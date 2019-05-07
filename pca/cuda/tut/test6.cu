#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>

#define N 256

__global__ void add(int *a, int *b, int* d) {
	int tx = threadIdx.x, ty = threadIdx.y;
	int bidx = blockIdx.x, bidy = blockIdx.y;
	int bx = blockDim.x, by = blockDim.y;
	int gy = gridDim.y;
	
	int bid = bidx*gy + bidy;
	int tid = bid*bx*by + tx*by + ty;
	
	int c = a[tid] + b[tid];
	atomicAdd(d, c);
}

void print_five(int* a, int* b){
	int r = 35;
	for(int i=0; i<10; ++i){
		r += i;	
		printf("%d %d\n", a[r], b[r]);
	}
}

void random_ints(int *a, int n){
   int i;
   for (i = 0; i < n; ++i)
       a[i] = rand() %10;
}

int main(void) {
	int *a, *b, *d; // host copies of a, b, c
	int *d_a, *d_b, *d_d; // device copies of a, b, c
	int size = N * sizeof(int), s = sizeof(int);
	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_d, s);
	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a, N);
	b = (int *)malloc(size); random_ints(b, N);
	d = (int *)malloc(s); *d = 0;

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_d, d, s, cudaMemcpyHostToDevice);
	// Launch add() kernel on GPU with N blocks
	
	dim3 thread(4,4),block(4, 4);

	add<<<block, thread>>>(d_a, d_b, d_d);
	// Copy result back to host
	cudaMemcpy(d, d_d, s, cudaMemcpyDeviceToHost);
	print_five(a,b);

	printf("%d\n",*d);

	// Cleanup
	free(a); free(b); free(d);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_d);
	return 0;
}
