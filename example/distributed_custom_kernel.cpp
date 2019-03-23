#include <gofmm_mpi.hpp>
#include <containers/KernelMatrix.hpp>

using namespace std;
using namespace hmlp;

template<typename T, typename TP>
T element_relu( const void* param, const TP* x, const TP* y, size_t d )
{
  return std::max( T(0), kernel_s<T, TP>::innerProduct( x, y, d ) );  
};

template<typename T, typename TP>
void matrix_relu( const void* param, const TP* X, const TP* Y, size_t d, T* K, size_t m, size_t n )
{
  kernel_s<T, TP>::innerProducts( X, Y, d, K, m, n );
  #pragma omp parallel for
  for ( size_t i = 0; i < m * n; i ++ ) K[ i ] = std::max( T(0), K[ i ] );
}

int main(int argc, char*argv[] )
{
    try
    {
        using T = float;
        size_t n = 20000;
        size_t m = 128;
        size_t k = 64;
        size_t s = 128;
        T stol = 1E-5;
        T budget = 0.01;
        size_t nrhs = 10;
        T lambda = 1.0;
        int provided;

        mpi::Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        if (provided != MPI_THREAD_MULTIPLE) exit(1);
        
        mpi::Comm CommGOFMM;
        mpi::Comm_dup(MPI_COMM_WORLD, &CommGOFMM);
        int comm_rank, comm_size;
        mpi::Comm_size(CommGOFMM, &comm_size);
        mpi::Comm_rank(CommGOFMM, &comm_rank);
        
        HANDLE_ERROR(hmlp_init(&argc, &argv, CommGOFMM));

        gofmm::Configuration<T> config(GEOMETRY_DISTANCE, n, m, k, s, stol, budget, false);
        int d = 10;

        DistData<STAR, CBLK, T> sources(d, n, CommGOFMM);
        sources.randn();

        kernel_s<T, T> kernel;
        kernel.type = USER_DEFINE;
        kernel.user_element_function = element_relu<T, T>;
        kernel.user_matrix_function = matrix_relu<T, T>;
        DistKernelMatrix<T, T> K(n, d, kernel, sources, CommGOFMM);
        
        mpigofmm::randomsplit<DistKernelMatrix<T, T>, 2, T> rkdtsplitter(K);
        mpigofmm::centersplit<DistKernelMatrix<T, T>, 2, T> splitter(K);
        
        auto neighborList = mpigofmm::FindNeighbors(K,rkdtsplitter, config, CommGOFMM);
        auto* tree_ptr = mpigofmm::Compress(K, neighborList, splitter, rkdtsplitter, config, CommGOFMM);
        
        auto& tree = *tree_ptr;

        DistData<RIDS, STAR, T> weights(n, nrhs, tree.treelist[0]->gids, CommGOFMM);
        weights.randn();

        auto u = mpigofmm::Evaluate(tree, weights);
    }
    catch( const exception & e)
    {
        cout << e.what() <<endl;
        return -1;
    }
    return 0;
};


    


