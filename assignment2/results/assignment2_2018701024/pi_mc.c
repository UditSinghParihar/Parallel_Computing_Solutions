#include <stdio.h>
#include <omp.h>
#include "random.h"

static long num_trials = 100000000;

int main(int argc, char const *argv[]){
	long Ncirc = 0;
	double pi, time;
	double r = 1.0;
	
	time = omp_get_wtime();
	#pragma omp parallel
	{
		#pragma omp single
		printf("Total threads: %d ",omp_get_num_threads());
		seed(-r, r);
		#pragma omp for reduction(+ : Ncirc)
			for(long i=0;i<num_trials; i++){
				double x = drandom();
				double y = drandom();
				double test = x*x + y*y;

				if (test <= r*r)
					Ncirc++;
			}
	}
	pi = 4.0 * ((double)Ncirc/(double)num_trials);
	double elapsed = omp_get_wtime()-time;

	printf("\n %ld trials, pi is %lf ",num_trials, pi);
	printf(" in %lf seconds\n",elapsed);
	
	return 0;
}