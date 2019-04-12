#include <gofmm.hpp>
/** Use abstracted virtual matrices. */
#include <containers/KernelMatrix.hpp>
/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;


template<typename T, typename TP>
T custom_element_kernel( const void* param, const TP* x, const TP* y, size_t d )
{
//   kernel_s inner product is defined in :gofmm/frame/containers/KernelMatrix.hpp
        return std::max( T(0), kernel_s<T, TP>::innerProduct( x, y, d ) );
};
   
template<typename T, typename TP>
void custom_matrix_kernel( const void* param, const TP* X, const TP* Y, size_t d, T* K, size_t m, size_t n )
{
   kernel_s<T, TP>::innerProducts( X, Y, d, K, m, n );
   #pragma omp parallel for
   for ( size_t i = 0; i < m * n; i ++ ) K[ i ] = std::max( T(0), K[ i ] );
}
    
   
             
