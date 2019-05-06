#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int* a, int* b, int* c){
	*c = *a + *b;
	//printf("c = %d\n", *c);
	printf("Thread id: %d\n", threadIdx.x);
	printf("Block id:%d\n", blockIdx.x);
}

int main(void) {
	int a, b, c; // host copies of a, b, c
	int *d_a, *d_b, *d_c; // device copies of a, b, c
	int size = sizeof(int);
	// Allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	// Setup input values
	a = 2;
	b = 7;

	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
	// Launch add() kernel on GPU
	add<<<3,5>>>(d_a, d_b, d_c);
	cudaDeviceSynchronize();
	// Copy result back to host
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
	//printf("a+b = %d\n", c);
	// Cleanup
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}
