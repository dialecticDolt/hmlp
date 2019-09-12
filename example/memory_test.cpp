/** Use MPI-GOFMM templates. */
#include <gofmm_mpi.hpp>
/** Use dense SPD matrices. */
#include <containers/SPDMatrix.hpp>
/** Use implicit kernel matrices (only coordinates are stored). */
#include <containers/KernelMatrix.hpp>

#include<iostream>
#include<fstream>
#include<unistd.h>

/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;

//Reports memory usage in kbytes
void process_mem_usage(double& vm_usage, double& resident_set)
{
    vm_usage     = 0.0;
    resident_set = 0.0;

    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;
}


//Default version of evaluate in hmlp/develop
void evaluate_memory_usage(int n_matvecs, int problem_size, int leaf_node_size, int neighbors, int maximum_rank, float stol, float budget, size_t nrhs, int argc, char* argv[]){
  try
  {
    using T = float;
	double vm, rss;
    /** Use float as data type. */
    /** [Required] Problem size. */
    size_t n = problem_size;
    /** Maximum leaf node size (not used in neighbor search). */
    size_t m = leaf_node_size;
    /** [Required] Number of nearest neighbors. */
    size_t k = neighbors;
    /** Maximum off-diagonal rank (not used in neighbor search). */
    size_t s = maximum_rank;

    /** MPI (Message Passing Interface): create a specific comm for GOFMM. */
    mpi::Comm CommGOFMM;
    mpi::Comm_dup( MPI_COMM_WORLD, &CommGOFMM );
    int comm_rank, comm_size;
    mpi::Comm_size( CommGOFMM, &comm_size );
    mpi::Comm_rank( CommGOFMM, &comm_rank );

    size_t n_loc = n / comm_size;
    size_t n_cut = n % comm_size;
    if (comm_rank < n_cut) n_loc++;
    
    /** [Step#0] HMLP API call to initialize the runtime. */
    HANDLE_ERROR( hmlp_init( &argc, &argv, CommGOFMM ) );

    /** [Step#1] Create a configuration for kernel matrices. */
    gofmm::Configuration<T> config( GEOMETRY_DISTANCE, n, m, k, s, stol, budget );
    
    /** [Step#2] Create a Gaussian kernel matrix with random 6D data. */
    size_t d = 5;
    
    /** Create local random point cloud. */
    Data<T> X_local( d, n_loc ); X_local.randn();
    
    /** Create distributed random point cloud. */
    DistData<STAR, CBLK, T> X( d, n, X_local, CommGOFMM );
    DistKernelMatrix<T, T> K( X, CommGOFMM );
    
    /** [Step#3] Create randomized and center splitters. */
    mpigofmm::randomsplit<DistKernelMatrix<T, T>, 2, T> rkdtsplitter( K );
    mpigofmm::centersplit<DistKernelMatrix<T, T>, 2, T> splitter( K );
    
    /** [Step#4] Perform the iterative neighbor search. */
    auto neighbors = mpigofmm::FindNeighbors(K, rkdtsplitter, config, CommGOFMM );

    /** [Step#5] Compress the matrix with an algebraic FMM. */
    auto* tree_ptr = mpigofmm::Compress( K, neighbors, splitter, rkdtsplitter, config, CommGOFMM );
    auto& tree = *tree_ptr;

    /** [Step#6] Set up weights for approximate MATVEC. */
    DistData<RIDS, STAR, T> weight( n, nrhs, tree.getOwnedIndices(), CommGOFMM ); 
    weight.randn();
    
    /** Perform Approximate MATVEC T times **/
	for(int i = 0; i<n_matvecs; i++){
    	auto u = mpigofmm::Evaluate(tree, weight);
		process_mem_usage(vm, rss);
		cout<<"i: " << i << "; Virtual Memory Usage: " << vm << "; Resident Set Size: "<< rss << endl;
	}
    /** [Step#9] HMLP API call to terminate the runtime. */
    HANDLE_ERROR( hmlp_finalize() );
  }
  catch ( const exception & e )
  {
    cout << e.what() << endl;
  }
}


//Version of evaulate that explicitly calls delete on the result vector
void evaluate_pointer_memory_usage(int n_matvecs, int problem_size, int leaf_node_size, int neighbors, int maximum_rank, float stol, float budget, size_t nrhs, int argc, char* argv[]){
  try
  {
	double vm, rss;
    /** Use float as data type. */
    using T = float;
    /** [Required] Problem size. */
    size_t n = problem_size;
    /** Maximum leaf node size (not used in neighbor search). */
    size_t m = leaf_node_size;
    /** [Required] Number of nearest neighbors. */
    size_t k = neighbors;
    /** Maximum off-diagonal rank (not used in neighbor search). */
    size_t s = maximum_rank;

    /** MPI (Message Passing Interface): create a specific comm for GOFMM. */
    mpi::Comm CommGOFMM;
    mpi::Comm_dup( MPI_COMM_WORLD, &CommGOFMM );
    int comm_rank, comm_size;
    mpi::Comm_size( CommGOFMM, &comm_size );
    mpi::Comm_rank( CommGOFMM, &comm_rank );

    size_t n_loc = n / comm_size;
    size_t n_cut = n % comm_size;
    if (comm_rank < n_cut) n_loc++;


    /** [Step#0] HMLP API call to initialize the runtime. */
    HANDLE_ERROR( hmlp_init( &argc, &argv, CommGOFMM ) );

    /** [Step#1] Create a configuration for kernel matrices. */
    gofmm::Configuration<T> config( GEOMETRY_DISTANCE, n, m, k, s, stol, budget );
    
    /** [Step#2] Create a Gaussian kernel matrix with random 6D data. */
    size_t d = 5;
    
    /** Create local random point cloud. */
    Data<T> X_local( d, n_loc ); X_local.randn();
    
    /** Create distributed random point cloud. */
    DistData<STAR, CBLK, T> X( d, n, X_local, CommGOFMM );
    DistKernelMatrix<T, T> K( X, CommGOFMM );
    
    /** [Step#3] Create randomized and center splitters. */
    mpigofmm::randomsplit<DistKernelMatrix<T, T>, 2, T> rkdtsplitter( K );
    mpigofmm::centersplit<DistKernelMatrix<T, T>, 2, T> splitter( K );
    
    /** [Step#4] Perform the iterative neighbor search. */
    auto neighbors = mpigofmm::FindNeighbors(K, rkdtsplitter, config, CommGOFMM );

    /** [Step#5] Compress the matrix with an algebraic FMM. */
    auto* tree_ptr = mpigofmm::Compress( K, neighbors, splitter, rkdtsplitter, config, CommGOFMM );
    auto& tree = *tree_ptr;

    /** [Step#6] Set up weights for approximate MATVEC. */
    DistData<RIDS, STAR, T> weight( n, nrhs, tree.getOwnedIndices(), CommGOFMM ); 
    weight.randn();
    
    /** Perform Approximate MATVEC T times **/
	for(int i = 0; i<n_matvecs; i++){
    	auto u = mpigofmm::Evaluate_Pointer(tree, weight);
		process_mem_usage(vm, rss);
		cout<<"Before Dealloc -- i: " << i << "; Virtual Memory Usage: " << vm << "; Resident Set Size: "<< rss << endl;
		delete(u);
		process_mem_usage(vm, rss);
		cout<<"After Dealloc -- i: " << i << "; Virtual Memory Usage: " << vm << "; Resident Set Size: "<< rss << endl;
	}
    /** [Step#9] HMLP API call to terminate the runtime. */
    HANDLE_ERROR( hmlp_finalize() );
  }
  catch ( const exception & e )
  {
    cout << e.what() << endl;
  }
}



int main( int argc, char *argv[] )
{

    int  provided;
    mpi::Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
    if ( provided != MPI_THREAD_MULTIPLE ) exit( 1 );
    
    int nTrials = 50;
	int problem_size = 10000;
	int leaf_node_size = 128;
	int neighbors = 128;
	int maximum_rank = 128;
	float stol = 1E-3;
	float budget = 0;
	size_t nrhs = 1;

    double vm, rss;
	//Start memory test
	process_mem_usage(vm, rss);
	cout<<"Initial -- Virtual Memory Usage: " << vm << "; Resident Set Size: "<< rss << endl;
  	evaluate_memory_usage(nTrials, problem_size, leaf_node_size, neighbors, maximum_rank, stol, budget, nrhs, argc, argv);
	evaluate_pointer_memory_usage(nTrials, problem_size, leaf_node_size, neighbors, maximum_rank, stol, budget, nrhs, argc, argv);
	process_mem_usage(vm, rss);
	cout<<"Final -- Virtual Memory Usage: " << vm << "; Resident Set Size: "<< rss << endl;

    /** Finalize Message Passing Interface. */
    mpi::Finalize();
    return 0;
}; /** end main() */
