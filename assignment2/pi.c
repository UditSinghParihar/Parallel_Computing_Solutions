#include <stdio.h>
#include <omp.h>
#include <time.h>

#define NUM_THREADS 4
#define steps 1000000000

void seq(void){
	double x=0.0, pi=0.0, sum = 0.0;
	double step = 1.0/(double) steps;
	for(int i=0; i<steps; i++){
		x = (i+0.5)*step;
		sum = sum + 4.0/(1.0+x*x);
	}
	pi = step * sum;
	printf("PI = %25.22g\n", pi);
}

void parallel(void){
	double step=0, pi=0, sum[NUM_THREADS];
	for(int i=0; i<NUM_THREADS; ++i)
		sum[i]=0.0;
	step = 1.0/(double)steps;

	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		for(int i=id; i<steps; i+=NUM_THREADS){
			double x = (i+0.5)*step;
			sum[id] += 4/(1.0+(x*x));
		}
	}

	for(int i=0; i<NUM_THREADS; ++i)
		pi += step*sum[i];
	printf("PI = %25.22g\n", pi);
}

void parallel_atomic(void){
	double step=0, pi=0;
	step = 1.0/(double)steps;

	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel
	{	
		double sum=0.0, x=0.0;
		int id = omp_get_thread_num();
		for(int i=id; i<steps; i+=NUM_THREADS){
			x = (i+0.5)*step;
			sum += 4/(1.0+(x*x));
		}

		#pragma omp atomic
		pi += sum*step;
	}

	printf("PI = %25.22g\n", pi);
}

void parallel_for(void){
	double step=0, pi=0;
	step = 1.0/(double)steps;

	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel
	{	
		double sum=0.0, x=0.0;
		#pragma omp for
		for(int i=0; i<steps; ++i){
			x = (i+0.5)*step;
			sum += 4/(1.0+(x*x));
		}

		#pragma omp atomic
		pi += sum*step;
	}

	printf("PI = %25.22g\n", pi);
}

void parallel_red(void){
	double step=0, pi=0;
	step = 1.0/(double)steps;

	double sum=0.0;
	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel for reduction(+ : sum)	
	for(int i=0; i<steps; ++i){
		double x = (i+0.5)*step;
		sum += 4/(1.0+(x*x));
	}
	pi += sum*step;

	printf("PI = %25.22g\n", pi);
}

int main(int argc, char const *argv[]){
	clock_t begin = clock();
	// seq();
	// parallel();
	// parallel_atomic();
	parallel_for();
	// parallel_red();
	
	clock_t end = clock();
	double time_spent = (double)(end-begin)/CLOCKS_PER_SEC;
	printf("Time taken: %g(second)\n", time_spent);

	return 0;
}