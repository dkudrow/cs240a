/* UCSB CS240A, Winter Quarter 2014
 * Main and supporting functions for the Conjugate Gradient Solver on a 5-point stencil
 *
 * NAMES: Daniel Kudrow, Victor, Joachim
 * PERMS: 8666141      , 
 * DATE: 2014-01-28
 */

#include "mpi.h"
#include "hw2harness.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// these should be accessible everywhere
int rank;
int size;

// we don't really need these
double* load_vec( char* filename, int* k );
void save_vec( int k, double* x );

// 
// matvec
//
// performs 'v = A * w' where
// A is an n x n matrix
// v and w are n-dimensional vectors
//
void matvec(double *v ,double *w, int k)
{
	int i, r, s;
	int start_i = rank ? k : 0;
	int rows_in_chunk = k / size;
	int node_elements_size = rows_in_chunk * k;
	double *extended_w;

	// allocate space for neighboring elements
	if (rank == 0 || rank == size-1)
		extended_w = malloc((k*k/size+k) * sizeof(double));
	else
		extended_w = malloc((k*k/size+2*k) * sizeof(double));

	// exchange elements with neighbors
	communicate_neighbor(extended_w, w, k);

	// matvec logic
	for (i = start_i; i<start_i+node_elements_size; i++) {
		r = rank * rows_in_chunk + (i-start_i) / k;
		s = i % k;
		v[i-start_i] = 4 * extended_w[i];
		if (r != 0)
			v[i-start_i] -= extended_w[i-k];
		if (s != 0)
			v[i-start_i] -= extended_w[i-1];
		if (s != k-1)
			v[i-start_i] -= extended_w[i+1];
		if (r != k-1)
			v[i-start_i] -= extended_w[i+k];
	}

	// cleanup
	free(extended_w);
	return v;
} 

//
// communicate_neighbor
//
// exchanges neighboring rows with the processors that own them
//
void communicate_neighbor(double * portion, double * v_part, int k)
{
	int i;
	int start_i = rank ? k : 0;
	int rows_in_chunk = (k) / size;
	MPI_Status status;

	// exchange relevent elments with neighbors
	if (rank == 0) {
		MPI_Send(&v_part[(rows_in_chunk-1)*k] , k , MPI_DOUBLE, rank+1, rank+1/*tag*/, MPI_COMM_WORLD);
		MPI_Recv(&portion[rows_in_chunk*k], k, MPI_DOUBLE, rank+1, rank, MPI_COMM_WORLD,&status);
	} else if (rank < size -1) {
		MPI_Send(&v_part[0] , k , MPI_DOUBLE, rank-1, rank-1/*tag*/, MPI_COMM_WORLD);
		MPI_Send(&v_part[(rows_in_chunk-1)*k] , k , MPI_DOUBLE, rank+1, rank+1/*tag*/, MPI_COMM_WORLD);
		MPI_Recv(&portion[0], k, MPI_DOUBLE, rank-1, rank, MPI_COMM_WORLD,&status);
		MPI_Recv(&portion[(rows_in_chunk+1)*k], k, MPI_DOUBLE, rank+1, rank, MPI_COMM_WORLD,&status);
	} else {
		MPI_Send(&v_part[0] , k , MPI_DOUBLE, rank-1, rank-1/*tag*/, MPI_COMM_WORLD);
		MPI_Recv(&portion[0], k, MPI_DOUBLE, rank-1, rank, MPI_COMM_WORLD,&status);
	}

	for (i=start_i; i<start_i+(k*k/size); i++)
		portion[i] = v_part[i-start_i];
}

//
// print_vec
//
// print a vector to stdout
//
void print_vec(double *vec, int n)
{
	int i;
	for (i=0; i<n; i++)
		printf("%1.2e ", vec[i]);
	printf("\n");
}

// 
// ddot
//
// returns the scalar dot product of n-dimensional vectors v_vec and w_vec
//
double ddot(double* v_vec, double *w_vec, int n)
{
	int i;
	double sendbuf = 0;
	double ret = 0;

	// compute partial sum
	for (i=0; i<n; i++)
		sendbuf += v_vec[i] * w_vec[i];

	// reduce results at root and broadcats to other nodes
	if (rank != 0) {
		MPI_Gather(&sendbuf, 1, MPI_DOUBLE, NULL, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&ret, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		return ret;
	} else {
		double *recvbuf = (double *)malloc(size * sizeof(double));

		MPI_Gather(&sendbuf, 1, MPI_DOUBLE, recvbuf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		for (i=0; i<size; i++)
			ret += recvbuf[i];

		MPI_Bcast(&ret, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		free(recvbuf);
		return ret;
	}
}

// 
// daxpy
//
// compute 'v = a*v + b*w' where
// a and b are scalars
// v and w are n-dimensional vectors
//
void daxpy(double *v_vec, double *w_vec, double a, double b, int n)
{
	int i;
	for (i=0; i<n; i++)
		v_vec[i] = a * v_vec[i] + b * w_vec[i];
}

//
// cgsolve
//
// CG solution to Poisson's Equation
//
double *cgsolve(int k)
{
	int i, first_i, last_i, part_size;
	int n = k * k;
	int maxiters = 1000 > 5*k ? 1000 : k;

	// partition the indices of b
	if (n % size) {
		first_i = (n / size + 1) * rank;
		last_i = (rank != size-1 ? first_i+n/size+1 : n);
	} else {
		first_i = n / size * rank;
		last_i = n / size * (rank + 1);
	}

	part_size = last_i - first_i;

	// allocate space for all vectors
	double *b_vec = (double *)malloc(part_size * sizeof(double));
	double *r_vec = (double *)malloc(part_size * sizeof(double));
	double *d_vec = (double *)malloc(part_size * sizeof(double));
	double *Ad_vec = (double *)malloc(part_size * sizeof(double));
	double *x_vec = (double *)malloc(part_size * sizeof(double));

	// fill the vectors with cs240_getB()
	for (i=0; i<part_size; i++) {
		double tmp = cs240_getB(first_i+i, n);
		b_vec[i] = tmp;
		r_vec[i] = tmp;
		d_vec[i] = tmp;
		x_vec[i] = 0;
	}

	// initialize looping conditions
	double normb = sqrt(ddot(b_vec, b_vec, part_size));
	double rtr = ddot(r_vec, r_vec, part_size);
	double relres = 1;

	// main loop
	i = 0;
	while (relres > 1e-6 && i++ < maxiters) {
		matvec(Ad_vec, d_vec, k);
		double alpha = rtr / ddot(d_vec, Ad_vec, part_size);
		daxpy(x_vec, d_vec, 1, alpha, part_size);
		daxpy(r_vec, Ad_vec, 1, -1*alpha, part_size);
		double rtrold = rtr;
		rtr = ddot(r_vec, r_vec, part_size);
		double beta = rtr / rtrold;
		daxpy(d_vec, r_vec, beta, 1, part_size);
		relres = sqrt(rtr) / normb;
	}

	// aggregate results in rank 0
	if (rank != 0) {
		MPI_Gather(x_vec, part_size, MPI_DOUBLE, NULL, part_size, MPI_DOUBLE,
				0, MPI_COMM_WORLD);
		return NULL;
	} else {
		double *results_buf = malloc(n * sizeof(double));
		MPI_Gather(x_vec, part_size, MPI_DOUBLE, results_buf, part_size,
				MPI_DOUBLE, 0, MPI_COMM_WORLD);
		return results_buf;
	}
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
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
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
	
	if (rank == 0) {
		printf("TEST: %s\n", cs240_verify(x, k, 0.0) ? "PASSED" : "FAILED");
	}

	if ( writeOutX ) {
		save_vec( k, x );
	}
		
	// Output
	printf( "Problem size (k): %d\n",k);
	if (niters>0){
          printf( "Norm of the residual after %d iterations: %lf\n",niters,norm);
        }
	printf( "Elapsed time during CGSOLVE: %lf\n", t2-t1);
	
        // Deallocate 
        if (niters > 0){
	  free(b);
	}
        if (niters > 0){
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

