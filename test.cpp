#include<iostream>
#include<mpi.h>

using namespace std;

int main(int argc, char *argv[]){
	int rank, size;
	char name[80];
	int length;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Get_processor_name(name, &length);


	cout << "Hello, MPI! Rank: "<< rank <<" size " <<size << " on "<<name<<endl;
	
	MPI_Finalize();
	return 1;
}

