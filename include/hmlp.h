#ifndef HMLP_H
#define HMLP_H

void hmlp_init();
void hmlp_finalize();



typedef enum
{
  HMLP_OP_N,
  HMLP_OP_T
} hmlpOperation_t;



void gkmx_dfma
(
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  double *A, int lda,
  double *B, int ldb,
  double *C, int ldc
);


void gkmx_dfma_simple
(
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  double *A, int lda,
  double *B, int ldb,
  double *C, int ldc
);


void gkmx_dconv_relu_pool
(
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  double *A, int lda,
  double *B, int ldb,
  double *C, int ldc
);

void dstrassen(
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  double *A, int lda,
  double *B, int ldb,
  double *C, int ldc
);







typedef enum
{
  KS_GAUSSIAN,
  KS_POLYNOMIAL,
  KS_LAPLACE,
  KS_GAUSSIAN_VAR_BANDWIDTH,
  KS_TANH,
  KS_QUARTIC,
  KS_MULTIQUADRATIC,
  KS_EPANECHNIKOV
} ks_type;

struct kernel_s
{
  ks_type type;
  double powe;
  double scal;
  double cons;
  double *hi;
  double *hj;
  double *h;
};

typedef struct kernel_s ks_t;

void dgsks
(
  ks_t *kernel,
  int m, int n, int k,
  double *u,             int *umap,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *w,             int *wmap
);

void dgsks_ref
(
  ks_t *kernel,
  int m, int n, int k,
  double *u,             int *umap,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *w,             int *wmap
);

void dgsknn
(
  int m, int n, int k, int r,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *D,             int *I
);

void dgsknn_ref
(
  int m, int n, int k, int r,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *D,             int *I
);

#ifdef HMLP_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <thrust/pair.h>


void dsq2nrm
(
  hmlpOperation_t transX, 
  int d, int n, 
  double* X2array[], const double* Xarray[], double* X, int ldx, 
  int batchSize 
);

void gkmm_dfma
(
  cudaStream_t stream,
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  const double *Aarray[], int lda,
  const double *Barray[], int ldb,
        double *Carray[], int ldc,
  int batchSize
);

void gkmm_dfma
(
  cudaStream_t stream,
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  const double *Aarray, int lda, int loa,
  const double *Barray, int ldb, int lob,
        double *Carray, int ldc, int loc,
  int batchSize
);

void gkrm_dkmean
(
  cudaStream_t stream, 
  hmlpOperation_t transA, hmlpOperation_t transB, 
  int m, int n, int k,
  double *Aarray[], double *A2array[], int lda,
  double *Barray[], double *B2array[], int ldb,
  thrust::pair<double,int>  *Carray[], int ldc, 
  int batchSize
);

void dstrassen
(
  cudaStream_t stream,
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  const double *Aarray[], int lda,
  const double *Barray[], int ldb,
        double *Carray[], int ldc,
  int batchSize
);

void dstrassen
(
  cudaStream_t stream,
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  const double *Aarray, int lda, int loa,
  const double *Barray, int ldb, int lob,
        double *Carray, int ldc, int loc,
  int batchSize
);
#endif // end ifdef HMLP_USE_CUDA

#endif // define HMLP_H
