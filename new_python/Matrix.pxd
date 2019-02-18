# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair

## Import dense SPDMatrix<T> from hmlp::SPDMatrix<T>.
cdef extern from "${CMAKE_SOURCE_DIR}/frame/containers/SPDMatrix.hpp" namespace "hmlp":
	cdef cppclass SPDMatrix[T]:
		# default constructor
		SPDMatrix() except +
		# size based constructor
		SPDMatrix( size_t m, size_t n ) except +

		# read from file
		void read( size_t m, size_t n, string &filename)

		# resize
		void resize( size_t m, size_t n )

		# get value
		T operator()(size_t i, size_t j) nogil

		# num rows
		size_t row()

		# num cols
		size_t col()

	## end cppclass SPDMatrix[T]
## end extern from.

cdef extern from "${CMAKE_SOURCE_DIR}/gofmm/tree.hpp" namespace "tree":
	cdef cppclass Tree:
		pass


cdef extern from "${CMAKE_SOURCE_DIR}/gofmm/gofmm.hpp" namespace "gofmm":
	cdef struct centersplit:
		SPDMatrix[float] *Kptr

	cdef struct randomsplit:
		SPDMatrix[float] *Kptr
	
## Import KernelMatrix from hmlp::KernelMatrix<T>
#cdef extern from "${CMAKE_SOURCE_DIR}/frame/containers/KernelMatrix.hpp" namespace "hmlp":
#    # first define kernel_s struct -- fully exposed so user can write own kernel func
#    cdef struct kernel_s[T,TP]:
#        # variables
#        kernel_type type # TODO is this bad practice?
#        T scal = 1 # gauss or sigmoid
#        T cons = 0 # for sigmoid
#
#        # inner products
#        @staticmethod
#        inline T innerProduct(const TP* x, constTP* y, size_t d)
#        @staticmethod
#        inline void innerProducts(const TP* X, constTP* Y, size_t d,
#                T* K, size_t m, size_t n))
#
#        # squared distances
#        @staticmethod
#        inline T squaredDistance(const TP* x, constTP* y, size_t d)
#        @staticmethod
#        inline void squaredDistances(const TP* X, constTP* Y, size_t d, 
#                T* K, size_t m, size_t n))
#
#
#        # operators
#        inline T operator() (const void* param, const TP* x, const TP* y, size_t d) const
#        inline void operator() (const void* param, const TP* X, 
#                const TP* Y, size_t d,t* K, size_t m, size_t n) const
#    
#    
#    cdef cppclass KernelMatrix[T]:
#        # symmetric constructor
#        KernelMatrix( size_t m_, size_t n_, size_t d_, kernel_s[T,T] &kernel_,
#                Data[T] & sources_)
#
#        # non-symmetric constructor
#        KernelMatrix( size_t m_, size_t n_, size_t d_, kernel_s[T,T] &kernel_,
#                Data[T] & sources_, Data[T] & targets_)
#         
#        # operator
#        T operator()(size_t i, size_t j)
#
#        # return dimension
#        size_t dim()
#
