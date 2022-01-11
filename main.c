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

void dgeqrf_(int * M, int *N, double *A, int *LDA,
             double *TAU, double *WORK, int *LWORK, int *INFO);

void dorgqr_(int *M, int *N, int *K, double *A, int *LDA,
            double *TAU, double *WORK, int *LWORK, int *INFO);

void pdgeqrf_(int *M, int *N, double *A, int *IA, int *JA, int *DESCA,
              double *TAU, double *WORK, int *LWORK, int *INFO);

void pdorgqr_(int *M, int *N, int *K, double *A, int *IA, int *JA, int *DESCA,
              double *TAU, double *WORK, int *LWORK, int *INFO);
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
   int nrowsTot = 1000;
   int ncolsTot = 1000;
   double **bcyclA;
   double *A, *ALoc;
   double *A_MPI, *Q_MPI, *R_MPI;

   mpi_err = MPI_Init(&argc, &argv);
   mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &procId);
   if(procId == 0)
      printf("running with %i processes\n", numprocs);

   int ncolsLoc = ncolsTot/numprocs;
   ALoc = (double *)malloc(nrowsTot*ncolsLoc*sizeof(double));
   if(procId == 0)
   {
      A_MPI = (double *)malloc(nrowsTot*ncolsTot*sizeof(double));
      Q_MPI = (double *)malloc(nrowsTot*ncolsTot*sizeof(double));
      R_MPI = (double *)malloc(ncolsTot*ncolsTot*sizeof(double));
      A = (double *)malloc(nrowsTot*ncolsTot*sizeof(double));
      for(int i = 0; i < nrowsTot; ++i)
         for(int j = 0; j < ncolsTot; ++j)
            A[j*nrowsTot+i] = (double)rand() / (double) RAND_MAX;

      {
         // Allocate copies so that identical problem will be solved with lapack
         double *A_copy = (double *)malloc(nrowsTot*ncolsTot*sizeof(double));
         memcpy(A_copy, A, nrowsTot*ncolsTot*sizeof(double));

         int INFO;
         int LDA   = nrowsTot;
         int LWORK = 2*ncolsTot;
         double *WORK = (double *)malloc(LWORK*sizeof(double));
         double *TAU = (double *)malloc(ncolsTot*sizeof(double));
         double *R = (double *)malloc(ncolsTot*ncolsTot*sizeof(double));

         double LP_START = get_cur_time();
         dgeqrf_(&nrowsTot, &ncolsTot, A_copy, &LDA,
                 TAU, WORK, &LWORK, &INFO);

         // extract R
         memset(R, 0, ncolsTot*ncolsTot*sizeof(double));
         for(int j = 0; j < ncolsTot; ++j)
            for(int i = 0; i <= j; ++i)
               R[j*ncolsTot+i] = A_copy[j*nrowsTot+i];

         // extract Q
         dorgqr_(&nrowsTot, &ncolsTot, &ncolsTot, A_copy, &LDA, TAU, WORK, &LWORK, &INFO);
         double LP_END = get_cur_time();
         printf("lapack wall time = %.6e\n", LP_END-LP_START);

//         for(int j = 0; j < ncolsTot; ++j)
//            for(int i = 0; i < nrowsTot; ++i)
//               printf("AQR[%i][%i] = %lf\n", i, j, A_copy[j*nrowsTot+i]);
//         printf("\n");

         // test A_Test = Q*R
         double *A_Test = (double *)malloc(nrowsTot*ncolsTot*sizeof(double));
         memset(A_Test, 0, nrowsTot*ncolsTot*sizeof(double));
         for(int i = 0; i < nrowsTot; ++i)
            for(int j = 0; j < ncolsTot; ++j)
               for(int k = 0; k < ncolsTot; ++k)
                  A_Test[j*nrowsTot+i] += A_copy[k*nrowsTot+i]*R[j*ncolsTot+k];
//         for(int j = 0; j < ncolsTot; ++j)
//            for(int i = 0; i < nrowsTot; ++i)
//               printf("ATest[%i][%i] = %lf\n", i, j, A_Test[j*nrowsTot+i]);
//         printf("\n");

         free(A_Test);
         free(R);
         free(TAU);
         free(WORK);
         free(A_copy);
      }

      bcyclA = (double**)malloc(numprocs*sizeof(double*));
      for(int i = 0; i < numprocs; ++i)
         bcyclA[i] = (double *)malloc(nrowsTot*ncolsLoc*sizeof(double));

      for(int pid = 0; pid < numprocs; ++pid)
         for(int j = 0; j < ncolsLoc; ++j)
            for(int i = 0; i < nrowsTot; ++i)
               bcyclA[pid][j*nrowsTot+i] = A[(j*numprocs+pid)*nrowsTot+i];

         for(int j = 0; j < ncolsLoc; ++j)
            for(int i = 0; i < nrowsTot; ++i)
            ALoc[j*nrowsTot+i] = bcyclA[0][j*nrowsTot+i];

      for(int pid = 1; pid < numprocs; ++pid)
         MPI_Send(&bcyclA[pid][0], nrowsTot*ncolsLoc, MPI_DOUBLE, pid, pid, MPI_COMM_WORLD);
//      for(int j = 0; j < ncolsTot; ++j)
//         for(int i = 0; i < nrowsTot; ++i)
//            printf("AGlob[%i][%i] = %lf\n", i, j, A[j*nrowsTot+i]);
//      printf("\n");

//      for(int pid = 0; pid < numprocs; ++pid)
//         for(int i = 0; i < nrowsTot; ++i)
//            for(int j = 0; j < ncolsLoc; ++j)
//               printf("bcyclA[%i][%i][%i]= %lf\n", pid, i, j, bcyclA[pid][i*ncolsLoc+j]);
   }
   if(procId != 0)
      MPI_Recv(&ALoc[0], nrowsTot*ncolsLoc, MPI_DOUBLE, 0, procId, MPI_COMM_WORLD, &status);

   int iam, nprocs;
   blacs_pinfo_(&iam, &nprocs);

   // nprocs x 1 processor layout
   int nprow = 1;
   int npcol = nprocs;
   char layout = 'C';
   int zero = 0;
   int ictxt, rowLoc, colLoc;
   blacs_get_(&zero, &zero, &ictxt);
   blacs_gridinit_(&ictxt, &layout, &nprow, &npcol);
   blacs_gridinfo_(&ictxt, &nprow, &npcol, &rowLoc, &colLoc);

   int mb = nrowsTot;
   int nb = ncolsTot/nprocs;
   int izero = 0;
   int mpA   = numroc_(&nrowsTot, &mb, &rowLoc, &izero, &nprow);
   int nqA   = numroc_(&ncolsTot, &nb, &colLoc, &izero, &npcol);
//   printf("mpA = %i\n", mpA);
//   printf("nqA = %i\n", nqA);
//   printf("rowLoc = %i\n", rowLoc);
//   printf("colLoc = %i\n", colLoc);
//   printf("row blocking factor = %i\n", mb);
//   printf("column blocking factor = %i\n", nb);
//   for(int pid = 0; pid < numprocs; ++pid)
//   {
//      if(procId == pid)
//      {
//         printf("pid = %i\n", procId);
//         for(int i = 0; i < mpA; ++i)
//            for(int j = 0; j < nqA; ++j)
//               printf("ALoc[%i][%i] = %lf\n", i, j, ALoc[i*ncolsLoc+j]);
//         printf("\n");
//      }
//      MPI_Barrier(MPI_COMM_WORLD);
//   }
   MPI_Barrier(MPI_COMM_WORLD);

   // 1 rhs for each process
   izero = 0;
   int info;
   int lddA = mpA;
   int descA[9];
   izero = 0;
   descinit_(descA, &nrowsTot, &ncolsTot, &mb, &nb,
              &izero, &izero, &ictxt, &lddA, &info);
   if(info != 0)
      printf("Error in descinit, info = %d\n", info);


   int IA = 1;
   int JA = 1;
   double *tauLoc = (double *)malloc(ncolsTot*sizeof(double));
   int lworkLoc = nrowsTot*ncolsTot*4;
   double *workLoc = (double *)malloc(lworkLoc*sizeof(double));
   int INFO   = 0;

   MPI_Barrier(MPI_COMM_WORLD);
   double MPI_start, MPI_end;
   if(procId == 0) { MPI_start = MPI_Wtime(); }
   pdgeqrf_(&nrowsTot, &ncolsTot, ALoc, &IA, &JA, descA,
         tauLoc, workLoc, &lworkLoc, &INFO);
   pdorgqr_(&nrowsTot, &ncolsTot, &nqA, ALoc, &IA, &JA, descA,
              tauLoc, workLoc, &lworkLoc, &INFO);
   MPI_Barrier(MPI_COMM_WORLD);
   if(procId == 0) {
      MPI_end = MPI_Wtime();
      printf("scalapack wall time: %.6e seconds\n\n", MPI_end-MPI_start);
   }
   MPI_Barrier(MPI_COMM_WORLD);

//   for(int pid = 0; pid < numprocs; ++pid)
//   {
//      if(procId == pid)
//      {
//         printf("pid = %i\n", procId);
//         for(int j = 0; j < ncolsLoc; ++j)
//            for(int i = 0; i < nrowsTot; ++i)
//               printf("AQRLoc[%i][%i] = %lf\n", i, j, ALoc[j*nrowsTot+i]);
//
//         for(int j = 0; j < ncolsLoc; ++j)
//         {
//            double dot = 0.0;
//            for(int i = 0; i < nrowsTot; ++i)
//               dot += ALoc[j*nrowsTot+i]*ALoc[j*nrowsTot+i];
//            printf("dotLoc = %lf\n", dot);
//         }
//         printf("\n");
//      }
//      MPI_Barrier(MPI_COMM_WORLD);
//   }

   free(ALoc);
   free(workLoc);
   free(tauLoc);
   if(procId == 0)
   {
      for(int i = 0; i < numprocs; ++i)
         free(bcyclA[i]);
      free(bcyclA);
      free(A);
      free(Q_MPI);
      free(R_MPI);
      free(A_MPI);
   }

   blacs_gridexit_(&ictxt);
   mpi_err = MPI_Finalize();
   return EXIT_SUCCESS;
}
