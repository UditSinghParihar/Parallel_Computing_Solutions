#include <stdio.h>
#include <omp.h>

#define size 4

void foo(void){
	int me =0;
	#ifdef _OPENMP
		me = omp_get_thread_num(); 
	#endif
		printf("Hello from thread: %d\n", me);
}

void foo_call(void){
	omp_set_num_threads(size);
	#pragma omp parallel
	{
		#pragma omp for
		for(int i=0; i<size; ++i){
			foo();
		}
	}
}

void sum(void){
	int len=20, arr[len], sum=0, tot=0;
	for(int i=0; i<len; ++i){
		arr[i] = 2*i;
		tot += arr[i];
	}

	omp_set_num_threads(size);
	#pragma omp parallel
	{
		#pragma omp for
		for(int i=0; i<len; ++i){
			#pragma omp atomic
			sum += arr[i];
		}
	}
	printf("Sum is: %d\nTotal is: %d\n", sum, tot);
	
}

int main(int argc, char const *argv[]){
	// foo_call();
	// sum();

	omp_set_num_threads(size);
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		printf("Hello from thread: %d\n", id);

		#pragma omp barrier
		if(id == 0){
			int nthreads = omp_get_num_threads();
			printf("There are %d threads\n", nthreads);
		}
	}

	return 0;
}