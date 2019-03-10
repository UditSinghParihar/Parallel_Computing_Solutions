#include<stdio.h>
#include<omp.h>
#include <time.h>

int main() {
	const int REPS = 1000000;
	int i;
	double balance = 0.0;
	printf("\nYour starting bank account balance is %0.2f\n", balance);
	
	clock_t begin = clock();

	// #pragma omp parallel for 	// A1
	// #pragma omp parallel for private(balance)	// B1
	#pragma omp parallel for reduction(+ : balance)
	for(i = 0; i < REPS; i++){
		// #pragma omp atomic	// C1
		balance += 10.0;
	}
	
	
	
	printf("\nAfter %d $10 deposits, your balance is %0.2f\n",
	REPS, balance);
	
	// #pragma omp parallel for 	// A2
	// #pragma omp parallel for private(balance)	// B2
	#pragma omp parallel for reduction(- : balance)
	for(i = 0; i < REPS; i++){
		// #pragma omp atomic	// C2
		balance -= 10.0;
	}

	clock_t end = clock();
	double time_spent = (double)(end-begin)/CLOCKS_PER_SEC;

	printf("\nAfter %d $10 withdrawals, your balance is %0.2f\n\n",
	REPS, balance);
	printf("Time taken: %g(second)\n", time_spent);
	
	return 0;
}