/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *  
 *  Copyright (C) 2014-2017, The University of Texas at Austin
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/  






// GKMX templates
#include <gkmx.hpp>

// Armv7a
#include <rank_k_d4x4.hpp>

using namespace hmlp::gkmx;


template<typename T>
struct identity 
{
  inline T operator()( const T& x, int i, int j, int b ) const 
  //inline T operator()( const T& x ) const 
  {
    return x; 
  }
  T** A2;
  T** B2;
};

template<typename TC, typename TV>
struct downcast
{
  inline TC operator()( const TV& x, int i, int j, int b ) const 
  //inline TC operator()( const TV& x ) const 
  {
    return (TC)x;
  }
  TV** A2;
  TV** B2;
};

void gkmx_dfma
(
  hmlpOperation_t transA, hmlpOperation_t transB,
	int m, int n, int k,
	double *A, int lda,
  double *B, int ldb,
  double *C, int ldc
)
{
  // Armv7a
  rank_k_asm_d4x4 semiringkernel;
  rank_k_asm_d4x4 microkernel;

  gkmx
  <192, 4096, 256, 4, 4, 192, 4096, 4, 4, 16,
  false,
  rank_k_asm_d4x4, rank_k_asm_d4x4,
  double, double, double, double>
  (
    transA, transB,
    m, n, k,
    A, lda,
    B, ldb,
    C, ldc,
    semiringkernel,
    microkernel
  );
};

void gkmx_sfma
(
  hmlpOperation_t transA, hmlpOperation_t transB,
	int m, int n, int k,
	float *A, int lda,
  float *B, int ldb,
  float *C, int ldc
)
{
  // Armv7a
  rank_k_asm_s4x4 semiringkernel;
  rank_k_asm_s4x4 microkernel;

  gkmx
  <192, 4096, 256, 4, 4, 192, 4096, 4, 4, 16,
  false,
  rank_k_asm_s4x4, rank_k_asm_s4x4,
  float, float, float, float>
  (
    transA, transB,
    m, n, k,
    A, lda,
    B, ldb,
    C, ldc,
    semiringkernel,
    microkernel
  );
};

void gkmx_dfma_simple
(
  hmlpOperation_t transA, hmlpOperation_t transB,
	int m, int n, int k,
	double *A, int lda,
  double *B, int ldb,
  double *C, int ldc
)
{
  std::plus<double> op1;
  std::multiplies<double> op2;

  identity<double> opkernel;

  double initV = 0.0;

  gkmm
  <104, 4096, 256, 8, 4, 104, 4096, 8, 4, 32,
  false>
  (
    transA, transB,
    m, n, k,
    A, lda,
    B, ldb,
    C, ldc,
    opkernel, op1, op2, initV
  );
};

void gkmx_mixfma_simple
(
  hmlpOperation_t transA, hmlpOperation_t transB,
	int m, int n, int k,
	double *A, int lda,
  double *B, int ldb,
  float  *C, int ldc
)
{
  std::plus<double> op1;
  std::multiplies<double> op2;

  //identity<double> opkernel;
  downcast<float, double> opkernel;

  double initV = 0.0;

  //gkmm
  //<104, 4096, 256, 8, 4, 104, 4096, 8, 4, 32,
  //false>
  //(
  //  transA, transB,
  //  m, n, k,
  //  A, lda,
  //  B, ldb,
  //  C, ldc,
  //  opkernel, op1, op2, initV
  //);
};

