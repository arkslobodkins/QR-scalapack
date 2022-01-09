#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "get_time.h"
#include "mpi.h"

// solves simple linear system using least squares pdgels routine
// for A*x = b, where A is a diagonal matrix with A[i][i] = i+1, and b[i] = 1.
// A is nxn, where n is a scalar multiple of numprocs. Each process
// computer n/numprocs rows, since numprocsx1 layout is used in this example.
// Entries of the solution should be x[i] = 1.0/(i+1)

void blacs_pinfo_(int*, int*);
void blacs_get_(int*, int*, int*);
void blacs_gridinit_(int*, char*, int*, int*);
void blacs_gridinfo_(int*, int*, int*, int*, int*);
void blacs_gridexit_(int*);
int numroc_(int*, int*, int*, int*, int*);
void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);

void dgels_(char *TRANS, int *M, int *N, int* nrhs,
            double* A, int* LDA,
            double* B, int* LDB,
            double *WORK, int *LWORK, int *INFO);

void pdgels_(char *trans, int *M, int *N, int *NRHS,
             double *A, int *IA, int *JA, int *DESCA,
             double *B, int *IB, int *JB, int *DESCB,
             double *WORK, int *LWORK, int *INFO);

double compute_norm(int len, double *v)
{
   double norm = 0.0;
   for(int i = 0; i < len; ++i)
      norm += v[i]*v[i];

   return sqrt(norm);
}

int main(int argc, char *argv[])
{
   int mpi_err, numprocs, procId, tag;
   MPI_Status status;
   int nrowsTot = 3000;
   int ncolsTot = 2000;
   double **cyclicArray, **cyclicVec;
   double *A, *b, *cyclicLoc, *bLoc;

   mpi_err = MPI_Init(&argc, &argv);
   mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &procId);
   if(procId == 0)
      printf("running with %i processes\n", numprocs);

   int locRows = nrowsTot/numprocs;
   cyclicLoc = (double *)malloc(locRows*ncolsTot*sizeof(double));
   bLoc = (double *)malloc(locRows*sizeof(double));
   if(procId == 0)
   {
      A = (double *)malloc(nrowsTot*ncolsTot*sizeof(double));
      for(int i = 0; i < nrowsTot; ++i)
         for(int j = 0; j < ncolsTot; ++j)
            A[j*nrowsTot+i] = (double)rand() / (double) RAND_MAX;

      b = (double *)malloc(nrowsTot*sizeof(double));
      for(int i = 0; i < nrowsTot; ++i)
         b[i] = (double)rand() / (double) RAND_MAX;

      {
         // Allocate copies so that identical problem will be solved with lapack
         double *A_copy = (double *)malloc(nrowsTot*ncolsTot*sizeof(double));
         memcpy(A_copy, A, nrowsTot*ncolsTot*sizeof(double));

         double *b_copy = (double *)malloc(nrowsTot*sizeof(double));
         memcpy(b_copy, b, nrowsTot*sizeof(double));

         char TRANS = 'N';
         int INFO;
         int LDA   = nrowsTot;
         int LEAD_DIM = nrowsTot>ncolsTot ? nrowsTot:ncolsTot;
         int NRHS = 1;
         int LWORK = 5*nrowsTot;
         double *WORK = (double *)malloc(LWORK*sizeof(double));
         double LP_START = get_cur_time();
         dgels_(&TRANS, &nrowsTot, &ncolsTot, &NRHS,
               A_copy, &LDA,
               b_copy, &LEAD_DIM,
               WORK, &LWORK, &INFO);
         double LP_END = get_cur_time();
         double la_sol_norm = compute_norm(ncolsTot, b_copy);

//         for(int j = 0; j < ncolsTot; ++j)
//            printf("lsq lapack solution: x[%i] = %.16e\n", j, b_copy[j]);

         printf("lapack wall time = %.6e\n", LP_END-LP_START);
         printf("lapack sol norm = %lf\n", la_sol_norm);
         free(WORK);
         free(A_copy);
         free(b_copy);
      }
      cyclicArray = (double**)malloc(numprocs*sizeof(double*));
      cyclicVec = (double**)malloc(numprocs*sizeof(double*));
      for(int i = 0; i < numprocs; ++i)
         cyclicArray[i] = (double *)malloc(locRows*ncolsTot*sizeof(double));
      for(int i = 0; i < numprocs; ++i)
         cyclicVec[i] = (double *)malloc(locRows*sizeof(double));

      for(int pid = 0; pid < numprocs; ++pid)
      {
         for(int i = 0; i < locRows; ++i)
            for(int j = 0; j < ncolsTot; ++j)
               cyclicArray[pid][i*ncolsTot+j] = A[(i*numprocs+pid)*ncolsTot+j];

         for(int i = 0; i < locRows; ++i)
            cyclicVec[pid][i] = b[i*numprocs+pid];
      }

      for(int i = 0; i < locRows; ++i)
         for(int j = 0; j < ncolsTot; ++j)
            cyclicLoc[i*ncolsTot+j] = cyclicArray[0][i*ncolsTot+j];
      for(int i = 0; i < locRows; ++i)
         bLoc[i] = cyclicVec[0][i];

      for(int pid = 1; pid < numprocs; ++pid)
         MPI_Send(&cyclicArray[pid][0], locRows*ncolsTot, MPI_DOUBLE, pid, pid, MPI_COMM_WORLD);
      for(int pid = 1; pid < numprocs; ++pid)
         MPI_Send(&cyclicVec[pid][0], locRows, MPI_DOUBLE, pid, 2*pid, MPI_COMM_WORLD);
//      for(int i = 0; i < nrowsTot; ++i)
//         for(int j = 0; j < ncolsTot; ++j)
//            printf("AGlob[%i][%i] = %lf\n", i, j, A[i*ncolsTot+j]);
//      for(int i = 0; i < nrowsTot; ++i)
//            printf("bGlob[%i] = %lf\n", i, b[i]);
   }
   for(int pid = 1; pid < numprocs; ++pid)
   if(procId != 0)
   {
      MPI_Recv(&cyclicLoc[0], locRows*ncolsTot, MPI_DOUBLE, 0, pid, MPI_COMM_WORLD, &status);
      MPI_Recv(&bLoc[0], locRows, MPI_DOUBLE, 0, 2*pid, MPI_COMM_WORLD, &status);
   }

   int iam, nprocs;
   blacs_pinfo_(&iam, &nprocs);

   // nprocs x 1 processor layout
   int nprow = nprocs;
   int npcol = 1;
   char layout = 'R';
   int zero = 0;
   int ictxt, rowLoc, colLoc;
   blacs_get_(&zero, &zero, &ictxt);
   blacs_gridinit_(&ictxt, &layout, &nprow, &npcol);
   blacs_gridinfo_(&ictxt, &nprow, &npcol, &rowLoc, &colLoc);

   // distribute rows over 2 processes, 8/nprocs x 8 matrix
   int mb = (nrowsTot+nprocs-1)/nprocs;
   int nb = ncolsTot;
   int izero = 0;
   int mpA   = numroc_(&nrowsTot, &mb, &rowLoc, &izero, &nprow);
   int nqA   = numroc_(&ncolsTot, &nb, &colLoc, &izero, &npcol);
//   printf("mpA = %i\n", mpA);
//   printf("nqA = %i\n", nqA);
//   printf("rowLoc = %i\n", rowLoc);
//   printf("colLoc = %i\n", colLoc);
//   printf("row blocking factor = %i\n", mb);
//   printf("column blocking factor = %i\n", nb);
   MPI_Barrier(MPI_COMM_WORLD);

   // 1 rhs for each process
   int NRHS   = 1;
   izero = 0;
   int nqrhs = numroc_(&NRHS, &mb, &colLoc, &izero, &npcol);
//   printf("nqrhs = %i\n", nqrhs);

   int info;
   int lddA = mpA;
   int descA[9];
   izero = 0;
   descinit_(descA, &nrowsTot, &ncolsTot, &mb, &nb,
              &izero, &izero, &ictxt, &lddA, &info);
   if(info != 0)
      printf("Error in descinit, info = %d\n", info);

   int descRHS[9];
   int ione = 1;
   izero = 0;
   descinit_(descRHS, &nrowsTot, &ione, &mb, &ione,
              &izero, &izero, &ictxt, &lddA, &info);
   if(info != 0)
      printf("Error in descinit, info = %d\n", info);



   char trans = 'N';
   double* ALoc = (double *)malloc(mpA*ncolsTot*sizeof(double));
   memcpy(ALoc, cyclicLoc, ncolsTot*locRows*sizeof(double));
//   for(int i = 0; i < mpA; ++i)
//      for(int j = 0; j < ncolsTot; ++j)
//         printf("ALoc[%i][%i] = %lf\n", i, j, ALoc[i*ncolsTot+j]);

   double *LSQ = (double *)malloc(mpA*sizeof(double));
   memcpy(LSQ, bLoc, locRows*sizeof(double));
//   for(int j = 0; j < mpA; ++j)
//      printf("rhsLoc[%i] = %lf\n", j, LSQ[j]);

   int IA = 1;
   int JA = 1;
   int IRHS = 1;
   int JRHS = 1;
   int LWORK = nrowsTot*ncolsTot*4;
   double *WORK = (double *)malloc(LWORK*sizeof(double));
   int INFO   = 0;

   MPI_Barrier(MPI_COMM_WORLD);
   double MPI_start, MPI_end;
   if(procId == 0) MPI_start = MPI_Wtime();
   pdgels_(&trans, &nrowsTot, &ncolsTot, &NRHS,
           ALoc, &IA, &JA, descA,
           LSQ, &IRHS, &JRHS, descRHS,
           WORK, &LWORK, &INFO);
   MPI_Barrier(MPI_COMM_WORLD);
   if(procId == 0) {
      MPI_end = MPI_Wtime();
      printf("scalapack wall time: %.6e seconds\n", MPI_end-MPI_start);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   if(procId == 0)
   {
      double scalapack_norm = compute_norm(ncolsTot, LSQ);
      printf("scalapack sol norm = %lf\n", scalapack_norm);
//      for(int j = 0; j < mpA; ++j)
//         printf("lsq solution: x[%i] = %.16e\n", j, LSQ[j]);
   }
//   MPI_Barrier(MPI_COMM_WORLD);
//   if(procId == 1)
//      for(int j = 0; j < mpA; ++j)
//         printf("lsq solution: x[%i] = %.16e\n", j, LSQ[j]);
//   MPI_Barrier(MPI_COMM_WORLD);
//   if(procId == 2)
//      for(int j = 0; j < mpA; ++j)
//         printf("lsq solution: x[%i] = %.16e\n", j, LSQ[j]);
//   MPI_Barrier(MPI_COMM_WORLD);
//   if(procId == 3)
//      for(int j = 0; j < mpA; ++j)
//         printf("lsq solution: x[%i] = %.16e\n", j, LSQ[j]);

   free(cyclicLoc);
   free(ALoc);
   free(LSQ);
   free(WORK);
   if(procId == 0)
   {
      for(int i = 0; i < numprocs; ++i)
         free(cyclicArray[i]);
      free(cyclicArray);
      free(A);
      free(b);
   }

   blacs_gridexit_(&ictxt);
   mpi_err = MPI_Finalize();
   return EXIT_SUCCESS;
}
