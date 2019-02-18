# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from Data cimport Data
from Config cimport Configuration

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
	cpdef cppclass Tree:
		pass


cdef extern from "${CMAKE_SOURCE_DIR}/gofmm/gofmm.hpp" namespace "gofmm":
	cpdef cppclass dTree_t:
		pass
	cpdef cppclass sTree_t:
		pass

	cdef struct centersplit:
		SPDMatrix[float]* Kptr

	cdef struct randomsplit:
		SPDMatrix[float]* Kptr
	
	#Try: Working with prototype *Compress
	sTree_t  *Compress(SPDMatrix[float]*, float, float, int, int, int)
	
	#TODO: Need SETUP and NODETYPE templates?
	#This one is so heavily templated I'm not sure how to handle it
	#Tree *Compress(SPDMatrix[float]*, Data[float], centersplit, randomsplit, Configuration[float]*)
	#This might be the better way
	#Tree *Compress(SPDMatrix[float]*, Data[pair[float, int]], centersplit, randomsplit, Configuration[float]*)
