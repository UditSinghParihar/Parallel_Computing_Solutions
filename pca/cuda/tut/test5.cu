#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>

#define N 256

__global__ void add(int *a, int *b, int *c, int* d) {
	int tx = threadIdx.x, ty = threadIdx.y;
	int bidx = blockIdx.x, bidy = blockIdx.y;
	int bx = blockDim.x, by = blockDim.y;
	int gx = gridDim.x, gy = gridDim.y;
	
	int bid = bidx*gy + bidy;	
	int tid = bid*bx*by + tx*by + ty;
	
	c[tid] = a[tid] + b[tid];
	atomicAdd(d, c[tid]);
}

void print_five(int* a, int* b, int* c){
	int r = 35;
	for(int i=0; i<10; ++i){
		r += i;	
		printf("%d %d %d\n", a[r], b[r], c[r]);
	}
}

void random_ints(int *a, int n){
   int i;
   for (i = 0; i < n; ++i)
       a[i] = rand() %10;
}

int main(void) {
	int *a, *b, *c, *d; // host copies of a, b, c
	int *d_a, *d_b, *d_c, *d_d; // device copies of a, b, c
	int size = N * sizeof(int), s = sizeof(int);
	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	cudaMalloc((void **)&d_d, s);
	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a, N);
	b = (int *)malloc(size); random_ints(b, N);
	c = (int *)malloc(size);
	d = (int *)malloc(s); *d = 0;

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_d, d, s, cudaMemcpyHostToDevice);
	// Launch add() kernel on GPU with N blocks
	
	dim3 thread(4,4),block(4, 4);

	add<<<block, thread>>>(d_a, d_b, d_c, d_d);
	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(d, d_d, s, cudaMemcpyDeviceToHost);
	print_five(a,b,c);

	printf("%d\n",*d);

	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}
