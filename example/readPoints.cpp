#include <gofmm_mpi.hpp>
#include <algorithm>
#include <containers/SPDMatrix.hpp>
#include <containers/KernelMatrix.hpp>
#include <cstring>
#include <iostream>
using namespace std;
using namespace hmlp;


int main(int argc, char *argv[]){
    try{

        //Set parameters

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

        //Initialize MPI
        int provided;
        mpi::Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        if( provided != MPI_THREAD_MULTIPLE) exit(1);
        mpi::Comm CommGOFMM;
        mpi::Comm_dup(MPI_COMM_WORLD, &CommGOFMM);
        int comm_rank, comm_size;
        mpi::Comm_size(CommGOFMM, &comm_size);
        mpi::Comm_rank(CommGOFMM, &comm_rank);

        //Initialize HMLP Runtime
        HANDLE_ERROR( hmlp_init(&argc, &argv, CommGOFMM) ); 

        //Read the data into a Column Block Dist Data object
        DistData<STAR, CBLK, T> X(d, n, CommGOFMM, filename) ;
    
        gofmm::Configuration<T> config(GEOMETRY_DISTANCE, n, m ,k, s, stol, budget, true);

        DistKernelMatrix<T, T> K(X, CommGOFMM);

        mpigofmm::randomsplit<DistKernelMatrix<T, T>, 2, T> rkdtsplitter(K);
        mpigofmm::centersplit<DistKernelMatrix<T, T>, 2, T> splitter(K);

        auto neighbors = mpigofmm::FindNeighbors(K, rkdtsplitter, config, CommGOFMM);
        auto tree_ptr = mpigofmm::Compress(K, neighbors, splitter, rkdtsplitter, config, CommGOFMM);
        auto& tree = *tree_ptr;
        auto rids = tree.treelist[0]->gids;


        //Read over the local data to set class indicator vector
        int n_local = X.col(); 
        Data<T> C(n_local, nrhs, 0.0);
        
        for(int i = 0; i<n_local; ++i){
            if ( i < n_local/2 ){
                C.setvalue(i, 0, 1.0);
            }
            else {
                C.setvalue(i, 1, 1.0);
            }
        }

        DistData<RBLK, STAR, T> w_rblk(n, nrhs, C, CommGOFMM); 
        DistData<RIDS, STAR, T> w_rids(n, nrhs, rids, CommGOFMM);
        w_rids = w_rblk;
        auto u = mpigofmm::Evaluate(tree, w_rids);

        //Store output values
        std::string fileout = "temp.data_r" + to_string(comm_rank)+".bin";
        u.write(fileout);
        
        std::string ridsout = "temp.gids_r" + to_string(comm_rank)+".bin";
        Data<size_t> rids_Data = Data<size_t>(rids.size(), 1, rids);
        rids_Data.write(ridsout);

        /** //Check local classes
        for(int j = 0; j<N_local; ++j){
            squareNorm = 0.0;
            for(int i=0; i<d;++i){
                temp = X.getvalue(i, j);
                squareNorm += temp*temp;
            }
            if(squareNorm > 10){
                C.setvalue(j, 0, 1.0);
            }
        }
        **/

        //Shutdown HMLP Runtime
        HANDLE_ERROR( hmlp_finalize() );

        //Shutdown MPI
        mpi::Finalize();

    }
    catch (const exception & e ){
        cout <<e.what() <<endl;
        return -1;
    }
}
