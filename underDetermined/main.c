#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "get_time.h"
#include "mpi.h"


// solves underdetermined system of equations using least squares approach using
// LAPACK and SCALAPACK // and compares their results and performance.
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
   int nrowsTot = 600;
   int ncolsTot = 1200;

   int rowsCols = nrowsTot*ncolsTot;
   double *b, *A, *ALoc, *RLoc;
   double *xLapack, *bLapack, *ALapack, *RLapack;
   double *Q_MPI, *L_MPI, *R_MPI, b_MPI;

   int nrowsTotTR = ncolsTot;
   int ncolsTotTR = nrowsTot;
   double *AT;

   mpi_err = MPI_Init(&argc, &argv);
   if(mpi_err != MPI_SUCCESS) exit(EXIT_FAILURE);
   mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   if(mpi_err != MPI_SUCCESS) exit(EXIT_FAILURE);
   mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &procId);
   if(mpi_err != MPI_SUCCESS) exit(EXIT_FAILURE);

   if(procId == 0)
      printf("running with %i MPI processes\n\n", numprocs);

   int ncolsLoc = ncolsTotTR/numprocs;
   ALoc = (double *)malloc(nrowsTotTR*ncolsLoc*sizeof(double));
   RLoc = (double *)malloc(ncolsTotTR*ncolsLoc*sizeof(double));
   if(procId == 0)
   {
      A = (double *)malloc(rowsCols*sizeof(double));
      b = (double *)malloc(nrowsTot*sizeof(double));
      for(int i = 0; i < rowsCols; ++i)
            A[i] = (double)rand() / (double) RAND_MAX;
      for(int i = 0; i < nrowsTot; ++i)
            b[i] = (double)rand() / (double) RAND_MAX;

      Q_MPI = (double *)malloc(rowsCols*sizeof(double));
      R_MPI = (double *)malloc(ncolsTot*ncolsTot*sizeof(double));
      L_MPI = (double *)malloc(ncolsTot*ncolsTot*sizeof(double));

      // Solve underdetermined least squares
      {
         double *ALapack = (double *)malloc(nrowsTot*ncolsTot*sizeof(double));
         memcpy(ALapack, A, nrowsTot*ncolsTot*sizeof(double));
         bLapack = (double *)malloc(ncolsTot*sizeof(double)); // underdetermined
         memcpy(bLapack, b, nrowsTot*sizeof(double));
         xLapack = (double *)malloc(ncolsTot*sizeof(double));

         char TRANS = 'N';
         int NRHS = 1;
         int LDA = nrowsTot;
         int LEAD_DIM = ncolsTot; // underdetermined system
         int LWORK = 5*ncolsTot;
         int INFO;
         double *WORK = (double *)malloc(LWORK*sizeof(double));
         double start = get_cur_time();
         dgels_(&TRANS, &nrowsTot, &ncolsTot, &NRHS, ALapack, &LDA, bLapack, &LEAD_DIM, WORK, &LWORK, &INFO);
         double end = get_cur_time();
         printf("lapack wall time: %.6e seconds\n", end - start);

         for(int i = 0; i < ncolsTot; ++i)
            xLapack[i] = bLapack[i]; // underdetermined

         double *b_Test = (double *)malloc(nrowsTot*sizeof(double));
         memset(b_Test, 0, nrowsTot*sizeof(double));
         for(int i = 0; i < nrowsTot; ++i)
            for(int j = 0; j < ncolsTot; ++j)
               b_Test[i] += A[j*nrowsTot+i]*xLapack[j];
         double XnormLapack = compute_norm(ncolsTot, xLapack);
         printf("x norm lapack = %.16e\n", XnormLapack);
         double lapackRes = computeRes(nrowsTot, b_Test, b);
         printf("lapack b residual = %.16e\n\n", lapackRes);

         free(ALapack);
         free(bLapack);
         free(xLapack);
         free(b_Test);
         free(WORK);
      }
      // transpose
      AT = (double *)malloc(rowsCols*sizeof(double));
      for(int i = 0; i < nrowsTot; ++i)
         for(int j = 0; j < ncolsTot; ++j)
            AT[i*ncolsTot+j] = A[j*nrowsTot+i];


   } // process 0
   double MPI_start_tot = MPI_Wtime();
   MPI_Scatter(AT, nrowsTotTR*ncolsLoc, MPI_DOUBLE,
               &ALoc[0], nrowsTotTR*ncolsLoc, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
//
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

   int mb = nrowsTotTR;
   int nb = ncolsTotTR/nprocs;
   int izero = 0;
   int mpA   = numroc_(&nrowsTotTR, &mb, &rowLoc, &izero, &nprow);
   int nqA   = numroc_(&ncolsTotTR, &nb, &colLoc, &izero, &npcol);

   izero = 0;
   int info;
   int lddA = mpA;
   int descA[9];
   izero = 0;
   descinit_(descA, &nrowsTotTR, &ncolsTotTR, &mb, &nb,
             &izero, &izero, &ictxt, &lddA, &info);
   if(info != 0)
      printf("Error in descinit, info = %d\n", info);

   int IA = 1;
   int JA = 1;
   int INFO = 0;
   int lworkLoc = nrowsTotTR*ncolsLoc*4;
   double *workLoc = (double *)malloc(lworkLoc*sizeof(double));
   double *tauLoc  = (double *)malloc(ncolsTotTR*sizeof(double));

   double MPI_start = MPI_Wtime();
   // compute QR
   pdgeqrf_(&nrowsTotTR, &ncolsTotTR, ALoc, &IA, &JA, descA,
            tauLoc, workLoc, &lworkLoc, &INFO);
   // extract RLoc
   for(int j = 0; j < ncolsLoc; ++j)
      for(int i = 0; i < ncolsTotTR; ++i)
         RLoc[j*ncolsTotTR+i] = ALoc[j*nrowsTotTR+i];
   pdorgqr_(&nrowsTotTR, &ncolsTotTR, &ncolsTotTR, ALoc, &IA, &JA, descA,
            tauLoc, workLoc, &lworkLoc, &INFO);
   double MPI_end = MPI_Wtime();
   if(procId == 0)
      printf("scalapack wall time: %.6e seconds\n", MPI_end-MPI_start);

   // Send Q to process 0
   MPI_Gather(&ALoc[0], nrowsTotTR*ncolsLoc, MPI_DOUBLE,
               Q_MPI, nrowsTotTR*ncolsLoc, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
   // Send R to process 0
   MPI_Gather(&RLoc[0], ncolsTotTR*ncolsLoc, MPI_DOUBLE,
               R_MPI, ncolsTotTR*ncolsLoc, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

   // compare results
   if(procId == 0)
   {
      for(int j = 0; j < ncolsTotTR; ++j)
         for(int i = 0; i < ncolsTotTR; ++i)
            L_MPI[j*ncolsTotTR+i] = R_MPI[i*ncolsTotTR+j];

      for(int j = 0; j < ncolsTotTR; ++j)
         for(int i = 0; i < j; ++i)
            L_MPI[j*ncolsTotTR+i] = 0.0;

      double *s_MPI = (double *)malloc(ncolsTotTR*sizeof(double));
      memcpy(s_MPI, b, ncolsTotTR*sizeof(double));

      //Solve Ls = b
      char UPLO  = 'L';
      char TRANS = 'N';
      char DIAG  = 'N';
      int NRHS   = 1;
      int INFO;
      dtrtrs_(&UPLO, &TRANS, &DIAG, &ncolsTotTR, &NRHS,
              L_MPI, &ncolsTotTR, s_MPI, &ncolsTotTR, &INFO);

      double *x_MPI = (double *)malloc(ncolsTot*sizeof(double));
      memset(x_MPI, 0, ncolsTot*sizeof(double));
      //Apply x = Qs
      for(int i = 0; i < ncolsTot; ++i)
         for(int j = 0; j < nrowsTot; ++j)
            x_MPI[i] += Q_MPI[j*nrowsTotTR+i]*s_MPI[j];
      double MPI_end_tot = MPI_Wtime();
      printf("total time of MPI solution = %.6e\n", MPI_end_tot - MPI_start_tot);

      double XnormScalapack = compute_norm(ncolsTot, x_MPI);
      printf("x norm scalapack = %.16e\n", XnormScalapack);
      double *b_Test_MPI = (double *)malloc(nrowsTot*sizeof(double));
      memset(b_Test_MPI, 0, nrowsTot*sizeof(double));
      for(int i = 0; i < nrowsTot; ++i)
         for(int j = 0; j < ncolsTot; ++j)
            b_Test_MPI[i] += A[j*nrowsTot+i]*x_MPI[j];
      double scalapackRes = computeRes(nrowsTot, b_Test_MPI, b);
      printf("scalapack b residual = %.16e\n\n", scalapackRes);

      free(b_Test_MPI);
      free(x_MPI);
      free(s_MPI);
      free(Q_MPI);
      free(R_MPI);
      free(L_MPI);
      free(A);
      free(b);
   }
   free(RLoc);
   free(ALoc);
   free(workLoc);
   free(tauLoc);

//   blacs_gridexit_(&ictxt);
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

