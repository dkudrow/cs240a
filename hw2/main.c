/* UCSB CS240A, Winter Quarter 2014
 * Main and supporting functions for the Conjugate Gradient Solver on a 5-point stencil
 *
 * NAMES:
 * PERMS:
 * DATE:
 */
/*#include "mpi.h"*/
#include <mpi/mpi.h>
#include "hw2harness.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double* load_vec( char* filename, int* k );
void save_vec( int k, double* x );

void print_vec(double *vec, int n)
{
	int i;
	for (i=0; i<n; i++)
		printf("%1.2e ", vec[i]);
	printf("\n");
}

double ddot(double* v_vec, double *w_vec, int n)
{
	int i;
	double ret = 0;
	for (i=0; i<n; i++)
		ret += v_vec[i] * w_vec[i];
	return ret;
}

void daxpy(double *v_vec, double *w_vec, double a, double b, int n)
{
	int i;
	for (i=0; i<n; i++)
		v_vec[i] = a * v_vec[i] + b * w_vec[i];
}

void matvec(double *A_vec, double *d_vec, int k)
{
	int r, s;
	for (r=0; r<k; r++) {
		for (s=0; s<k; s++) {
			int i = r * k + s;
			A_vec[i] = 4 * d_vec[i];
			if (r != 0)
				A_vec[i] -= d_vec[i-k];
			if (s != 0)
				A_vec[i] -= d_vec[i-1];
			if (s != k-1)
				A_vec[i] -= d_vec[i+1];
			if (r != k-1)
				A_vec[i] -= d_vec[i+k];
		}
	}
}

double *cgsolve(int k)
{
	int i;
	int n = k * k;
	int maxiters = 1000 > 5*k ? 1000 : k;

	double *b_vec = malloc(n * sizeof(double));
	double *r_vec = malloc(n * sizeof(double));
	double *d_vec = malloc(n * sizeof(double));
	double *A_vec = malloc(n * sizeof(double));
	double *x_vec = malloc(n * sizeof(double));

	for (i=0; i<n; i++) {
		double tmp = cs240_getB(i, n);
		b_vec[i] = tmp;
		r_vec[i] = tmp;
		d_vec[i] = tmp;
		x_vec[i] = 0;
	}

	double normb = sqrt(ddot(b_vec, b_vec, n));
	double rtr = ddot(r_vec, r_vec, n);
	double relres = 1;

	i = 0;
	while (relres > 1e-6 && i++ < maxiters) {

		matvec(A_vec, d_vec, k);
		double alpha = rtr / ddot(d_vec, A_vec, n);
		daxpy(x_vec, d_vec, 1, alpha, n);
		daxpy(r_vec, A_vec, 1, -1*alpha, n);
		double rtrold = rtr;
		rtr = ddot(r_vec, r_vec, n);
		double beta = rtr / rtrold;
		daxpy(d_vec, r_vec, beta, 1, n);
		relres = sqrt(rtr) / normb;
	}
	return x_vec;
}


int main( int argc, char* argv[] ) {
	int writeOutX = 0;
	int n, k;
	int maxiterations = 1000;
	int niters=0;
 	double norm;
	double* b;
	double* x;
	double time;
	double t1, t2;
	
	MPI_Init( &argc, &argv );
	
	// Read command line args.
	// 1st case runs model problem, 2nd Case allows you to specify your own b vector
	if ( argc == 3 ) {
		k = atoi( argv[1] );
		n = k*k;
		// each processor calls cs240_getB to build its own part of the b vector!
	} else if  ( !strcmp( argv[1], "-i" ) && argc == 4 ) {
		b = load_vec( argv[2], &k );
	} else {
		printf( "\nCGSOLVE Usage: \n\t"
			"Model Problem:\tmpirun -np [number_procs] cgsolve [k] [output_1=y_0=n]\n\t"
			"Custom Input:\tmpirun -np [number_procs] cgsolve -i [input_filename] [output_1=y_0=n]\n\n");
		exit(0);
	}
	writeOutX = atoi( argv[argc-1] ); // Write X to file if true, do not write if unspecified.

	
	// Start Timer
	t1 = MPI_Wtime();
	
	// CG Solve here!
	x = cgsolve(k);
 	// End Timer
	t2 = MPI_Wtime();
	
	printf("TEST: %s\n", cs240_verify(x, k, 0.0) ? "PASSED" : "FAILED");

	if ( writeOutX ) {
		save_vec( k, x );
	}
		
	// Output
	printf( "Problem size (k): %d\n",k);
	if(niters>0){
          printf( "Norm of the residual after %d iterations: %lf\n",niters,norm);
        }
	printf( "Elapsed time during CGSOLVE: %lf\n", t2-t1);
	
        // Deallocate 
        if(niters > 0){
	  free(b);
	}
        if(niters > 0){
          free(x);
	}
	
	MPI_Finalize();
	
	return 0;
}


/*
 * Supporting Functions
 *
 */

// Load Function
// NOTE: does not distribute data across processors
double* load_vec( char* filename, int* k ) {
	FILE* iFile = fopen(filename, "r");
	int nScan;
	int nTotal = 0;
	int n;
	
	if ( iFile == NULL ) {
		printf("Error reading file.\n");
		exit(0);
	}
	
	nScan = fscanf( iFile, "k=%d\n", k );
	if ( nScan != 1 ) {
		printf("Error reading dimensions.\n");
		exit(0);
	}
	
	n = (*k)*(*k);
	double* vec = (double *)malloc( n * sizeof(double) );
	
	do {
		nScan = fscanf( iFile, "%lf", &vec[nTotal++] );
	} while ( nScan >= 0 );
	
	if ( nTotal != n+1 ) {
		printf("Incorrect number of values scanned n=%d, nTotal=%d.\n",n,nTotal);
		exit(0);
	}
	
	return vec;
}

// Save a vector to a file.
void save_vec( int k, double* x ) { 
	FILE* oFile;
	int i;
	oFile = fopen("xApprox.txt","w");
	
	fprintf( oFile, "k=%d\n", k );
	
	for (i = 0; i < k*k; i++) { 
    	fprintf( oFile, "%lf\n", x[i]);
 	} 

	fclose( oFile );
}

