#include <stdio.h>
#include "mpi.h"

int main (int argc, char *argv[])
{ 
	 int rank, i;

 	  MPI_Init (&argc, &argv);

   MPI_Comm_rank (MPI_COMM_WORLD, &rank);


   if (rank == 0) i = 27;

   MPI_Bcast ((void *)&i, 1, MPI_INT, 0, MPI_COMM_WORLD);

   printf ("[%d] i = %d\n", rank, i);

   // Wait for every process to reach this code


   MPI_Barrier (MPI_COMM_WORLD);

   MPI_Finalize();
   return 0;


}

