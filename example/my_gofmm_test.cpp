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

/** Use MPI-GOFMM templates. */
#include <gofmm_mpi.hpp>
#include <algorithm>
/** Use dense SPD matrices. */
#include <containers/SPDMatrix.hpp>
/** Use implicit kernel matrices (only coordinates are stored). */
#include <containers/KernelMatrix.hpp>
/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;

/** 
 *  @brief In this example, we explain how you can compress generic
 *         SPD matrices and kernel matrices using MPIGOFMM. 
 */ 
int main( int argc, char *argv[] )
{
  try
  {
    /** Use float as data type. */
    using T = float;
    /** [Required] Problem size. */
    size_t n = 4000;
    /** Maximum leaf node size (not used in neighbor search). */
    size_t m = 128;
    /** [Required] Number of nearest neighbors. */
    size_t k = 64;
    /** Maximum off-diagonal rank (not used in neighbor search). */
    size_t s = 128;
    /** Approximation tolerance (not used in neighbor search). */
    T stol = 1E-5;
    /** The amount of direct evaluation (not used in neighbor search). */
    T budget = 0.01;
    /** Number of right-hand sides. */
    size_t nrhs = 2;
    /** Dimensionality of data */
    size_t d = 2;
    /** File to read in (created in Python script) */
    std::string filename = "points.bin";
    /** what rank to print from (set to -1 for no print) */
    size_t prank = 1;
    
    /** MPI (Message Passing Interface): check for THREAD_MULTIPLE support. */
    int provided;
    mpi::Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
    if ( provided != MPI_THREAD_MULTIPLE ) exit( 1 );
    /** MPI (Message Passing Interface): create a specific comm for GOFMM. */
    mpi::Comm CommGOFMM;
    mpi::Comm_dup( MPI_COMM_WORLD, &CommGOFMM );
    int comm_rank, comm_size;
    mpi::Comm_size( CommGOFMM, &comm_size );
    mpi::Comm_rank( CommGOFMM, &comm_rank );
    
    /** [Step#0] HMLP API call to initialize the runtime. */
    HANDLE_ERROR( hmlp_init( &argc, &argv, CommGOFMM ) );

    /** [Step#1] Create a configuration for kernel matrices. */
    gofmm::Configuration<T> config2( GEOMETRY_DISTANCE, n, m, k, s, stol, budget,true);
    
    /** [Step#2] Initialize w1 local vector and rblk  */
    // compute size info
    size_t n_loc = n / comm_size;
    size_t n_cut = n % comm_size;
    if ( comm_rank < n_cut ){n_loc ++;}

    // initialize local copy, set values
    vector<T> w1_local( n_loc * nrhs, 0.0 );
    size_t cur_gid = comm_rank;
    std::cout << " W BEGINS HERE " << std::endl;
    for(size_t wi = 0; wi < n_loc; wi++)
    {
        if (cur_gid < n/2)
        { 
            // belong to class 1
            w1_local[wi] = 1.0;
            //if(comm_rank == prank){std::cout << "Rank " << comm_rank << " set global point "<< cur_gid<<" (local "<< wi<<") to class 1" << std::endl;}
        }else
        {
            // belong to class 2
            w1_local[wi + n_loc] = 1.0;
            //if(comm_rank == prank){std::cout << "Rank " << comm_rank << " set global point "<< cur_gid<<" (local "<< wi<<") to class 2" << std::endl;}
        }
        cur_gid += comm_size;
    }

    // load into distdata
    DistData<RBLK, STAR, T> w1_rblk( n, nrhs, w1_local, CommGOFMM );

    /** [Step#3] Load random point cloud, and corresponding kernel and splitters */
    DistData<STAR, CBLK, T> X = DistData<STAR, CBLK, T>( d, n, CommGOFMM, filename ) ;
    DistKernelMatrix<T, T> K2( X, CommGOFMM );
    mpigofmm::randomsplit<DistKernelMatrix<T, T>, 2, T> rkdtsplitter2( K2 );
    mpigofmm::centersplit<DistKernelMatrix<T, T>, 2, T> splitter2( K2 );
    
    /** [Step#4] Compress the matrix with an algebraic FMM (after neighbor compute) */
    auto neighbors2 = mpigofmm::FindNeighbors( K2, rkdtsplitter2, config2, CommGOFMM );
    auto* tree_ptr2 = mpigofmm::Compress( K2, neighbors2, splitter2, rkdtsplitter2, config2, CommGOFMM );
    auto& tree2 = *tree_ptr2;

    /** [Step#5] Compute an approximate MATVEC. */
    auto rids = tree2.treelist[0]->gids;
    DistData<RIDS, STAR, T> w1( n, nrhs, tree2.treelist[ 0 ]->gids, CommGOFMM );
    w1 = w1_rblk;
    auto u1 = mpigofmm::Evaluate( tree2, w1 );

    /** [Step#6] Write to file (each process writes own file). */
    std::string fileout =  "temp.data_r" +to_string(comm_rank) + ".bin";
    u1.write(fileout);

    if (comm_rank == prank)
    {
        std::cout << "U STARTS HERE" << std::endl;
        for (size_t ui = 0; ui < n_loc; ui ++)
        {
            std::cout << "Rid: "<< rids[ui] << " c0: "<< u1[ui] <<" c1: " << u1[ui+n_loc]<< std::endl;
        }
    }
    std::string ridsout = "temp.gids_r" + to_string(comm_rank) + ".bin";
    Data<size_t> rids_Data = Data<size_t>(rids.size(),1,rids);
    rids_Data.write(ridsout);
        
    /** [Step#7] HMLP API call to terminate the runtime. */
    HANDLE_ERROR( hmlp_finalize() );
    /** Finalize Message Passing Interface. */
    mpi::Finalize();
  }
  catch ( const exception & e )
  {
    cout << e.what() << endl;
    return -1;
  }
  return 0;
}; /** end main() */
