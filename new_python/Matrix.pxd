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
		void randspd(T a, T b)

	## end cppclass SPDMatrix[T]
## end extern from.

cdef extern from "${CMAKE_SOURCE_DIR}/gofmm/tree.hpp" namespace "tree":
	cpdef cppclass Tree:
		pass


cdef extern from "${CMAKE_SOURCE_DIR}/gofmm/gofmm.hpp" namespace "hmlp::gofmm":
	cpdef cppclass dTree_t:
		pass
	cpdef cppclass sTree_t:
		pass
	
	cdef cppclass sSPDMatrix_t:
		sSPDMatrix_t() except +
		sSPDMatrix_t( size_t m, size_t n ) except +
		void resize(size_t m, size_t n)
		size_t row()
		size_t col()
		float getvalue(size_t i)
		void setvalue(size_t i, float v)
		void randspd()
	
	cdef cppclass dSPDMatrix_t:
		dSPDMatrix_t() except +
		dSPDMatrix_t( size_t m, size_t n ) except +
		void resize(size_t m, size_t n)
		size_t row()
		size_t col()
		double getvalue(size_t i)
		void setvalue(size_t i, double v)
		void randspd()

	cdef struct centersplit:
		SPDMatrix[float]* Kptr

	cdef struct randomsplit:
		SPDMatrix[float]* Kptr
	
	#Working prototype *Compress
	sTree_t *Compress(sSPDMatrix_t*, float, float)
	dTree_t *Compress(dSPDMatrix_t&, double, double)
	dTree_t *Compress(SPDMatrix[double]&, double, double)
	sTree_t *Compress(SPDMatrix[float]&, float, float)	
	#TODO: Add more configuration options (Take in config object). Write C++ code to apply configuration. 
	

	#This option is so heavily templated I'm not sure how to handle it. (Tree should be templated on setup and nodetype)
	#Tree *Compress(SPDMatrix[float]*, Data[float], centersplit, randomsplit, Configuration[float]*)
	#Tree *Compress(SPDMatrix[float]*, Data[pair[float, int]], centersplit, randomsplit, Configuration[float]*)

	
		
