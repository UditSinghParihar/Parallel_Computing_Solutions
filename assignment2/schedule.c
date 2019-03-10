#include <unistd.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#define THREADS 10
#define N 100000000

void seq(void){
	printf("Running %d iterations on %d threads sequentially.\n", N, THREADS);
	
	clock_t begin = clock();
	for(int i=0; i<N; i++){}
	clock_t end = clock();
	double time_spent = (double)(end-begin)/CLOCKS_PER_SEC;
	printf("All done in %g(second)\n", time_spent);
}

void dynamic(void){
	printf("Running %d iterations on %d threads dynamically.\n", N, THREADS);
	
	clock_t begin = clock();
	#pragma omp parallel for schedule(dynamic) num_threads(THREADS)
	for(int i=0; i<N; i++){}
	clock_t end = clock();
	double time_spent = (double)(end-begin)/CLOCKS_PER_SEC;
	printf("All done in %g(second)\n", time_spent);
}

void stat(void){
	printf("Running %d iterations on %d threads statically.\n", N, THREADS);
	
	clock_t begin = clock();
	#pragma omp parallel for schedule(static) num_threads(THREADS)
	for(int i=0; i<N; i++){}
	clock_t end = clock();
	double time_spent = (double)(end-begin)/CLOCKS_PER_SEC;
	printf("All done in %g(second)\n", time_spent);
}

void guided(void){
	printf("Running %d iterations on %d threads guided.\n", N, THREADS);
	
	clock_t begin = clock();
	#pragma omp parallel for schedule(guided) num_threads(THREADS)
	for(int i=0; i<N; i++){}
	clock_t end = clock();
	double time_spent = (double)(end-begin)/CLOCKS_PER_SEC;
	printf("All done in %g(second)\n", time_spent);
}

int main(int argc, char const *argv[]) {
	// seq();
	dynamic();
	// stat();
	// guided();
	
	return 0;
}