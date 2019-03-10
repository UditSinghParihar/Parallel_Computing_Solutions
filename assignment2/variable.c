#include <unistd.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#define THREADS 4
#define N 16

void guided(void){
	time_t begin = time(NULL);
	#pragma omp parallel for schedule(guided) num_threads(THREADS)
	for(int i = 0; i<N; i++){
		sleep(i);
		printf("Thread %d has completed iteration %d.\n", omp_get_thread_num(), i);
	}
	time_t end = time(NULL);
	printf("All done in %lu(second)\n", time(NULL)-begin);
}

void stat(void){
	time_t begin = time(NULL);
	#pragma omp parallel for schedule(static) num_threads(THREADS)
	for(int i = 0; i<N; i++){
		sleep(i);
		printf("Thread %d has completed iteration %d.\n", omp_get_thread_num(), i);
	}
	time_t end = time(NULL);
	printf("All done in %lu(second)\n", time(NULL)-begin);
}

void dynamic(void){
	time_t begin = time(NULL);
	#pragma omp parallel for schedule(dynamic) num_threads(THREADS)
	for(int i = 0; i<N; i++){
		sleep(i);
		printf("Thread %d has completed iteration %d.\n", omp_get_thread_num(), i);
	}
	time_t end = time(NULL);
	printf("All done in %lu(second)\n", time(NULL)-begin);
}

int main(int argc, char const *argv[]){
	// guided();
	// stat();
	dynamic();
	
	return 0;
}