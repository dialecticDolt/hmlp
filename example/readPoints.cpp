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
        using T = float;
        size_t N = 4000;
        size_t d = 2;
        string fileName = "points";

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
        DistData<STAR, CBLK, T> X(d, N, CommGOFMM, fileName) ;
    
        

        //Read over the local data to set class indicator vector
        /**
        int N_local = X.col(); 
        Data<T> C(N_local, 1, 0.0);
        T squareNorm = 0.0;
        T temp;

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
