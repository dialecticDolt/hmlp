/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *  
 *  Copyright (C) 2014-2018, The University of Texas at Austin
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

/** Use GOFMM templates. */
#include <gofmm.hpp>
/** Use dense SPD matrices. */
#include <containers/SPDMatrix.hpp>
/** Use implicit kernel matrices (only coordinates are stored). */
#include <containers/KernelMatrix.hpp>
/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;

/** 
 *  @brief In this example, we explain how you can compute
 *         approximate all-nearest neighbors (ANN) using GOFMM. 
 */ 
int main( int argc, char *argv[] )
{
  try
  {
    /** Use float as data type. */
    using T = float;
    /** [Required] Problem size. */
    size_t n = 1000000;
    /** Maximum leaf node size (not used in neighbor search). */
    size_t m = 512;
    /** [Required] Number of nearest neighbors. */
    size_t k = 64;
    /** Maximum off-diagonal rank (not used in neighbor search). */
    size_t s = 128;
    /** Approximation tolerance (not used in neighbor search). */
    T stol = 1E-5;
    /** The amount of direct evaluation (not used in neighbor search). */
    T budget = 0.01;

    HANDLE_ERROR( hmlp_init( &argc, &argv ) );
    /** [Step#1] Create a configuration for kernel matrices. */
    gofmm::Configuration<T> config2( GEOMETRY_DISTANCE, n, m, k, s, stol, budget );
    /** [Step#2] Create a Gaussian kernel matrix with random 6D data. */
    size_t d = 10;
    Data<T> X( d, n ); X.randn();
    KernelMatrix<T> K2( X );
    cout << "Number of rows: " << K2.row() << " number of columns: " << K2.col() << endl;
    cout << "K(0,0) " << K2( 0, 0 ) << " K(1,2) " << K2( 1, 2 ) << endl;
    /** [Step#3] Create a randomized splitter. */
    gofmm::randomsplit<KernelMatrix<T>, 2, T> rkdtsplitter2( K2 );
    /** [Step#4] Perform the iterative neighbor search. */
    auto neighbors2 = gofmm::FindNeighbors( K2, rkdtsplitter2, config2 );
    cout << "Number of neighboprs: " << neighbors2.row() << " number of queries: " << neighbors2.col() << endl;
    for ( int i = 0; i < std::min( k, (size_t)10 ); i ++ )
      printf( "[%E,%5lu]\n", neighbors2( i, 0 ).first, neighbors2( i, 0 ).second );

    /** [Step#5] HMLP API call to terminate the runtime. */
    HANDLE_ERROR( hmlp_finalize() );
  }
  catch ( const exception & e )
  {
    cout << e.what() << endl;
    return -1;
  }
  return 0;
}; /** end main() */
