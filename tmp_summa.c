/******************************************************************************************
 *
 *	Filename:	summa.c
 *	Purpose:	A paritally implemented program for MSCS6060 HW. Students will complete 
 *			the program by adding SUMMA implementation for matrix multiplication C = A * B.  
 *	Assumptions:    A, B, and C are square matrices n by n; 
 *			the total number of processors (np) is a square number (q^2).
 *	To compile, use 
 *	    mpicc -o summa summa.c
 *       To run, use
 *	    mpiexec -n $(NPROCS) ./summa
 *********************************************************************************************/

#include <stdio.h>
#include <time.h>	
#include <stdlib.h>	
#include <math.h>	
#include <string.h>
#include "mpi.h"

#define min(a, b) ((a < b) ? a : b)
#define SZ 4000		//Each matrix of entire A, B, and C is SZ by SZ. Set a small value for testing, and set a large value for collecting experimental data.


/**
 *   Allocate space for a two-dimensional array
 */
double **alloc_2d_double(int n_rows, int n_cols) {
	int i;
	double **array;
	array = (double **)malloc(n_rows * sizeof (double *));
	array[0] = (double *) malloc(n_rows * n_cols * sizeof(double));
	for (i=1; i<n_rows; i++){
		array[i] = array[0] + i * n_cols;
	}
	return array;
}

/**
 *	Initialize arrays A and B with random numbers, and array C with zeros. 
 *	Each array is setup as a square block of blck_sz.
 **/
void initialize(double **lA, double **lB, double **lC, int blck_sz){
	int i, j;
	double value;
	// Set random values...technically it is already random and this is redundant
	for (i=0; i<blck_sz; i++){
		for (j=0; j<blck_sz; j++){
			lA[i][j] = (double)rand() / (double)RAND_MAX;
			lB[i][j] = (double)rand() / (double)RAND_MAX;
			lC[i][j] = 0.0;
		}
	}
}

void initialize_test(double **lA, double **lB, double **lC, int block_sz, int coordinates[2]){
	int i, j, ii, jj;
	for(ii = 0; ii < block_sz; ii++) {
		for(jj = 0; jj < block_sz; jj++) {
			i = ii + block_sz * coordinates[0];
			j = jj + block_sz * coordinates[1];
			if (i == 0 && j==0 ) {	
				if (lC[ii][jj]!=1) {
					printf("C[%d][%d] is incorrect\n", ii,jj);
				}
			}else if(i == j){
				if (lC[ii][jj]!=2) {
					printf("C[%d][%d] is incorrect\n", ii,jj);
				}
			}else if( (i-1) == j){
				if (lC[ii][jj]!=1) {
					printf("C[%d][%d] is incorrect\n", ii,jj);
				}
			}else if(i == (j-1) ){
				if (lC[ii][jj]!=1) {
					printf("C[%d][%d] is incorrect\n", ii,jj);
				}
			}	
		}
	}
	printf("Test pass\n");
}



/**
 *	Perform the SUMMA matrix multiplication. 
 *       Follow the pseudo code in lecture slides.
 */

// basic SUMMA Algorithm
void basicSUMMA( double **my_C, double **my_A, double **my_B, int block_sz){
	for (int k = 0; k < block_sz; ++k) {
		for (int i = 0; i < block_sz; ++i) {
			for (int j = 0; j < block_sz; ++j) {
				my_C[i][j] += my_A[i][k] * my_B[k][j];
			}
		}
	}
}

void matmul(int my_rank, int proc_grid_sz, int block_sz, double **my_A,
		double **my_B, double **my_C, int coordinates[2] , MPI_Comm grid_comm){

	int remain_dims[2];
	// 1) setup new communicators for both rows and columns
	MPI_Comm row_comm, col_comm;

	// for row communicator
	remain_dims[0] = 0;
	remain_dims[1] = 1;

	// partition the communicator we mad earlier into subgroups, which form lower-dimensional Cartesian subgrids, input => communicator, remain_dims
	// outputs a communicator containing the subgrid, 
	MPI_Cart_sub(grid_comm, remain_dims, &row_comm);

	// for column communicator
	remain_dims[0] = 1;
	remain_dims[1] = 0;
	MPI_Cart_sub(grid_comm, remain_dims, &col_comm);

	// setup our 2 dimensional buffers to hold data of size block_sz * block_sz
	double **buffA, **buffB;
	buffA =  alloc_2d_double(block_sz, block_sz);
	buffB=  alloc_2d_double(block_sz, block_sz);

	int k;
	for(k=0; k<proc_grid_sz; k++){
		if(coordinates[1] == k){
			memcpy(buffA[0], my_A[0], block_sz*block_sz*sizeof(double));
		}
		// broadcast message from row_comm
		MPI_Bcast(*buffA, block_sz*block_sz, MPI_DOUBLE, k, row_comm);

		if (coordinates[0] == k) {
			memcpy(buffB[0], my_B[0], block_sz*block_sz*sizeof(double));
		}
		// broadcast message from col_comm
		MPI_Bcast(*buffB, block_sz*block_sz, MPI_DOUBLE, k, col_comm);

		if (coordinates[0] == k && coordinates[1] == k) {
			basicSUMMA(my_C, my_A, my_B, block_sz);
		}else if(coordinates[0] == k){
			basicSUMMA(my_C, buffA, my_B, block_sz);
		}else if(coordinates[1] == k){
			basicSUMMA(my_C, my_A, buffB, block_sz);
		}else{
			basicSUMMA(my_C, buffA, buffB, block_sz);
		}
	}
}


int main(int argc, char *argv[]) {
	int rank, num_proc;							//process rank and total number of processes
	double start_time, end_time, total_time;	// for timing
	int block_sz;								// Block size length for each processor to handle
	int proc_grid_sz;							// 'q' from the slides

	int wrap[2];
	int coordinates[2];
	int dim_sz[2];
	int reorder = 1;


	srand(time(NULL));							// Seed random numbers

	/* insert MPI functions to 
	 * 1) start process, 
	 * 2) get total number of processors and 
	 * 3) process rank*/

	// 1. initialize with the given command line arguments 
	MPI_Init(&argc, &argv); 

	// 2. get the rank of the calling process
	// MPI_COMM_WORLD is the defaeult communicator which groups 
	// all the processes when the program is started 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// 3. determine the number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);






	/* assign values to 1) proc_grid_sz and 2) block_sz*/
	// for a square matrix, proc_grid_size is the square root of number of 
	// processes
	// lets get the square root with a builtin function `sqrt`. 
	// As sqrt takes double and returns double, let's cast the number of 
	// processes
	// to double and cast back the result to integer
	proc_grid_sz = (int)sqrt((double)num_proc);

	// divide the matrix size by proc_gird_sz to get the block size
	block_sz = SZ/proc_grid_sz;

	dim_sz[0]=dim_sz[1] = proc_grid_sz;
	wrap[0] = wrap[1] = 1;

	// 2) Let's setup a communicator between these processes 

	printf("Initializing communicator \n");
	// initialize a new communicator to store our communicator
	MPI_Comm grid_comm;
	// create a new handle to a new communicator and store it on our  grid_comm object
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim_sz, wrap, reorder, &grid_comm);
	// get the rank of the calling process and store it in my_rank
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	// get process coords in Cartesian topology given rank in group
	MPI_Cart_coords(grid_comm, my_rank, 2, coordinates);

	if (SZ % proc_grid_sz != 0){
		printf("Matrix size cannot be evenly split amongst resources!\n");
		printf("Quitting....\n");
		exit(-1);
	}

	// Create the local matrices on each process

	double **A, **B, **C;
	A = alloc_2d_double(block_sz, block_sz);
	B = alloc_2d_double(block_sz, block_sz);
	C = alloc_2d_double(block_sz, block_sz);



	initialize(A, B, C, block_sz);


	//printf("STARTING SUMMA.......\n");
	// Use MPI_Wtime to get the starting time
	start_time = MPI_Wtime();


	// Use SUMMA algorithm to calculate product C
	matmul(rank, proc_grid_sz, block_sz, A, B, C, coordinates,grid_comm);


	//printf("ENDING SUMMA.......\n");
	// Use MPI_Wtime to get the finishing time
	end_time = MPI_Wtime();


	// Obtain the elapsed time and assign it to total_time
	total_time = end_time - start_time;

	printf("TOTAL TIME: %f ms \n",total_time);

	// Insert statements for testing
	//initialize_test(A, B, C, block_sz, coordinates);

	if (rank == 0){
		// Print in pseudo csv format for easier results compilation
		printf("squareMatrixSideLength,%d,numMPICopies,%d,walltime,%lf\n",
				SZ, num_proc, total_time);
	}

	// Destroy MPI processes
	MPI_Finalize();

	//printf("CLEARED EXEC ENV.");

	return 0;
}
