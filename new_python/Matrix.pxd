# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.string cimport string


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

cdef extern from "${CMAKE_SOURCE_DIR}/gofmm/gofmm.hpp" namespace "gofmm":
        cdef struct centersplit:
            SPDMatrix[float] *Kptr

        cdef struct randomsplit:
            SPDMatrix[float] *Kprt
