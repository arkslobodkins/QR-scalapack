#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "get_time.h"
#include "mpi.h"


// computes QR factorization using LAPACK and SCALAPACK routines and explicitly extracts Matrices Q and R.
// Thorough tests are performed to compare results and performance of SCALAPACK to that of LAPACK.
// Currently number of columns in a matrix must be a multiple of number of processors.

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

void dgels_(char *TRANS, int *M, int *N, int *NRHS,
            double *A, int *LDA, double *B, int *LDB,
            double *WORK, int *LWORK, int *INFO);

void dtrtrs_(char *UPLO, char *TRANS, char *DIAG, int *N, int *NRHS,
		       double *A, int *LDA, double *b, int *LDB, int *INFO);

void pdgeqrf_(int *M, int *N, double *A, int *IA, int *JA, int *DESCA,
              double *TAU, double *WORK, int *LWORK, int *INFO);

void pdorgqr_(int *M, int *N, int *K, double *A, int *IA, int *JA, int *DESCA,
              double *TAU, double *WORK, int *LWORK, int *INFO);

void PrintFull(int M, int N, double *A, const char *name);
void PrintUpper(int M, int N, double *A, const char *name);
double compute_norm(int len, double *v);
double computeRes(int len, double *A, double *B);

int main(int argc, char *argv[])
{
   int mpi_err, numprocs, procId, tag;
   MPI_Status status;
   int nrowsTot = 1000;
   int ncolsTot = 100;
   int rowsCols = nrowsTot*ncolsTot;
   double *b, *A, *ALoc, *RLoc;
   double *xLapack, *bLapack, *ALapack, *RLapack;
   double *A_MPI, *Q_MPI, *R_MPI;

   mpi_err = MPI_Init(&argc, &argv);
   if(mpi_err != MPI_SUCCESS) exit(EXIT_FAILURE);
   mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   if(mpi_err != MPI_SUCCESS) exit(EXIT_FAILURE);
   mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &procId);
   if(mpi_err != MPI_SUCCESS) exit(EXIT_FAILURE);

   if(procId == 0)
      printf("running with %i MPI processes\n\n", numprocs);

   int ncolsLoc = ncolsTot/numprocs;
   ALoc = (double *)malloc(nrowsTot*ncolsLoc*sizeof(double));
   RLoc = (double *)malloc(ncolsTot*ncolsLoc*sizeof(double));
   if(procId == 0)
   {
      A       = (double *)malloc(rowsCols*sizeof(double));
      b       = (double *)malloc(nrowsTot*sizeof(double));
      A_MPI = (double *)malloc(rowsCols*sizeof(double));
      Q_MPI = (double *)malloc(rowsCols*sizeof(double));
      R_MPI = (double *)malloc(rowsCols*sizeof(double));
      ALapack = (double *)malloc(rowsCols*sizeof(double));
      RLapack = (double *)malloc(ncolsTot*ncolsTot*sizeof(double));
      xLapack = (double *)malloc(ncolsTot*sizeof(double));
      bLapack = (double *)malloc(nrowsTot*sizeof(double));

      for(int i = 0; i < rowsCols; ++i)
            A[i] = (double)rand() / (double) RAND_MAX;
      memcpy(ALapack, A, rowsCols*sizeof(double));
      for(int i = 0; i < nrowsTot; ++i)
            b[i] = (double)rand() / (double) RAND_MAX;
      memcpy(bLapack, b, nrowsTot*sizeof(double));

      // Part 1: Compute and extract QR for later comparison
      {
         // Allocate copy so that identical problem will be solved with lapack

         int INFO;
         int LDA   = nrowsTot;
         int LWORK = 2*ncolsTot;
         double *WORK = (double *)malloc(LWORK*sizeof(double));
         double *TAU = (double *)malloc(ncolsTot*sizeof(double));

         double LP_START = get_cur_time();
         dgeqrf_(&nrowsTot, &ncolsTot, ALapack, &LDA,
                 TAU, WORK, &LWORK, &INFO);
         // extract R
         memset(RLapack, 0, ncolsTot*ncolsTot*sizeof(double));
         for(int j = 0; j < ncolsTot; ++j)
            for(int i = 0; i <= j; ++i)
               RLapack[j*ncolsTot+i] = ALapack[j*nrowsTot+i];
         // extract Q
         dorgqr_(&nrowsTot, &ncolsTot, &ncolsTot, ALapack, &LDA, TAU, WORK, &LWORK, &INFO);
         double LP_END = get_cur_time();
         printf("lapack wall time = %.6e\n", LP_END-LP_START);

         // test A_Test = Q*R
         double *A_Test = (double *)malloc(nrowsTot*ncolsTot*sizeof(double));
         memset(A_Test, 0, nrowsTot*ncolsTot*sizeof(double));
         for(int i = 0; i < nrowsTot; ++i)
            for(int j = 0; j < ncolsTot; ++j)
               for(int k = 0; k < ncolsTot; ++k)
                  A_Test[j*nrowsTot+i] += ALapack[k*nrowsTot+i]*RLapack[j*ncolsTot+k];
         double lapackRes = computeRes(rowsCols, A_Test, A);
         printf("lapack A residual = %.16e\n\n", lapackRes);
         free(A_Test);
         free(TAU);
         free(WORK);
      }
      // Part 2: solve least squares problem directly
      {
         double *ALSQ = (double *)malloc(nrowsTot*ncolsTot*sizeof(double));
         memcpy(ALSQ, A, nrowsTot*ncolsTot*sizeof(double));

         char TRANS = 'N';
         int NRHS = 1;
         int LDA = nrowsTot;
         int LEAD_DIM = nrowsTot;
         int LWORK = 5*ncolsTot;
         int INFO;
         double *WORK = (double *)malloc(LWORK*sizeof(double));
         dgels_(&TRANS, &nrowsTot, &ncolsTot, &NRHS, ALSQ, &LDA, bLapack, &LEAD_DIM, WORK, &LWORK, &INFO);
         for(int i = 0; i < ncolsTot; ++i)
            xLapack[i] = bLapack[i];

         free(ALSQ);
         free(bLapack);
         free(WORK);
      }
   } // process 0
   MPI_Scatter(A, nrowsTot*ncolsLoc, MPI_DOUBLE,
               &ALoc[0], nrowsTot*ncolsLoc, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

   int iam, nprocs;
   blacs_pinfo_(&iam, &nprocs);
   // 1 x nprocs processor layout
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
   int INFO = 0;
   int lworkLoc = nrowsTot*ncolsLoc*4;
   double *workLoc = (double *)malloc(lworkLoc*sizeof(double));
   double *tauLoc  = (double *)malloc(ncolsTot*sizeof(double));

   double MPI_start = MPI_Wtime();
   // compute QR
   pdgeqrf_(&nrowsTot, &ncolsTot, ALoc, &IA, &JA, descA,
            tauLoc, workLoc, &lworkLoc, &INFO);
   // extract RLoc
   for(int j = 0; j < ncolsLoc; ++j)
      for(int i = 0; i < ncolsTot; ++i)
         RLoc[j*ncolsTot+i] = ALoc[j*nrowsTot+i];
   pdorgqr_(&nrowsTot, &ncolsTot, &ncolsTot, ALoc, &IA, &JA, descA,
            tauLoc, workLoc, &lworkLoc, &INFO);
   double MPI_end = MPI_Wtime();
   if(procId == 0)
      printf("scalapack wall time: %.6e seconds\n", MPI_end-MPI_start);

   // Send Q to process 0
   MPI_Gather(&ALoc[0], nrowsTot*ncolsLoc, MPI_DOUBLE,
               Q_MPI, nrowsTot*ncolsLoc, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
   // Send R to process 0
   MPI_Gather(&RLoc[0], ncolsTot*ncolsLoc, MPI_DOUBLE,
               R_MPI, ncolsTot*ncolsLoc, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

   // compare results
   if(procId == 0)
   {
      for(int i = 0; i < nrowsTot*ncolsLoc; ++i)
         Q_MPI[i] = ALoc[i];
      for(int i = 0; i < ncolsTot*ncolsLoc; ++i)
         R_MPI[i] = RLoc[i];
      for(int j = 0; j < ncolsTot; ++j)
         for(int i = ncolsTot-1; i > j; --i)
            R_MPI[j*ncolsTot+i] = 0.0;

      memset(A_MPI, 0, rowsCols*sizeof(double));
      for(int i = 0; i < nrowsTot; ++i)
         for(int j = 0; j < ncolsTot; ++j)
            for(int k = 0; k <= j; ++k)
               A_MPI[j*nrowsTot+i] += Q_MPI[k*nrowsTot+i]*R_MPI[j*ncolsTot+k];

      double scalapackRes = computeRes(rowsCols, A_MPI, A);
      printf("scalapackRes A residual = %.16e\n\n", scalapackRes);

      double QnormLapack = compute_norm(nrowsTot*ncolsTot, ALapack);
      printf("Q norm lapack = %.16e\n", QnormLapack);
      double QnormScalapack = compute_norm(nrowsTot*ncolsTot, Q_MPI);
      printf("Q norm scalapack = %.16e\n", QnormScalapack);

      double RnormLapack = compute_norm(ncolsTot*ncolsTot, RLapack);
      printf("R norm lapack = %.16e\n", RnormLapack);
      double RnormScalapack = compute_norm(ncolsTot*ncolsTot, R_MPI);
      printf("R norm scalapack = %.16e\n", RnormScalapack);

      double *b_MPI = (double *)malloc(ncolsTot*sizeof(double));
      memset(b_MPI, 0, ncolsTot*sizeof(double));
      //Apply Q'b
      for(int j = 0; j < ncolsTot; ++j)
         for(int i = 0; i < nrowsTot; ++i)
            b_MPI[j] += Q_MPI[j*nrowsTot+i]*b[i];

      //Solve Rx = Q'b
      char UPLO  = 'U';
      char TRANS = 'N';
      char DIAG  = 'N';
      int NRHS   = 1;
      int INFO;
      dtrtrs_(&UPLO, &TRANS, &DIAG, &ncolsTot, &NRHS,
              R_MPI, &ncolsTot, b_MPI, &ncolsTot, &INFO);

      double XnormLapack = compute_norm(ncolsTot, xLapack);
      printf("x norm lapack = %.16e\n", XnormLapack);
      double XnormScalapack = compute_norm(ncolsTot, b_MPI);
      printf("x norm scalapack = %.16e\n", XnormScalapack);
      double xRes = computeRes(ncolsTot, xLapack, b_MPI);
      printf("infinity norm of their difference: %.16e\n", xRes);

      free(b_MPI);
      free(Q_MPI);
      free(R_MPI);
      free(A_MPI);
      free(RLapack);
      free(ALapack);
      free(xLapack);
      free(A);
      free(b);
   }
   free(RLoc);
   free(ALoc);
   free(workLoc);
   free(tauLoc);

   blacs_gridexit_(&ictxt);
   mpi_err = MPI_Finalize();
   return EXIT_SUCCESS;
}


void PrintUpper(int M, int N, double *A, const char *name)
{
   for(int j = 0; j < N; ++j)
      for(int i = 0; i <= j; ++i)
         printf("%s[%i][%i] = %lf\n", name, i, j, A[j*M+i]);
   printf("\n");
}

void PrintFull(int M, int N, double *A, const char *name)
{
   for(int j = 0; j < N; ++j)
      for(int i = 0; i < M; ++i)
         printf("%s[%i][%i] = %lf\n", name, i, j, A[j*M+i]);
   printf("\n");
}

double compute_norm(int len, double *v)
{
   double norm = 0.0;
   for(int i = 0; i < len; ++i)
      norm += v[i]*v[i];

   return sqrt(norm);
}

#define MAX(a, b) ((a) > (b) ? (a) : (b))
double computeRes(int len, double *A, double *B)
{
   double norm = 0.0;
   for(int i = 0; i < len; ++i)
      norm = MAX(norm, fabs(A[i]-B[i]));

   return norm;
}

