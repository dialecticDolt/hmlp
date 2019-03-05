# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from Data cimport Data
from Runtime cimport hmlpError_t
from libcpp cimport bool
from Config cimport Configuration, DistanceMetric

# useful for splitter templating
cdef extern from *:
    ctypedef int two "2"

cdef extern from *:
    ctypedef bool use_runtime "true"
    ctypedef bool use_opm_task "true"
    ctypedef bool nnprune "true"
    ctypedef bool cache "true"

cdef extern from *:
    ctypedef bool verdad "true"

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
        #hmlpError_t NeighborSearch(DistanceMetric, size_t, const vector[size_t]&, const vector[size_t]& , Data[pair[T, size_t]]]  
    ## end cppclass SPDMatrix[T]
## end extern from.

## Import KernelMatrix from hmlp::KernelMatrix<T>
cdef extern from "${CMAKE_SOURCE_DIR}/frame/containers/KernelMatrix.hpp" namespace "hmlp":
    # enum for kernel type
    ctypedef enum kernel_type:
        GAUSSIAN
        SIGMOID
        POLYNOMIAL
        LAPLACE
        GAUSSIAN_VAR_BANDWIDTH
        TANH
        QUARTIC
        MULTIQUADRATIC
        EPANECHNIKOV
        USER_DEFINE


    # first define kernel_s struct -- fully exposed so user can write own kernel func
    cdef cppclass kernel_s[T,TP]:
        # variables
        kernel_type type # TODO is this bad practice?
        T scal # gauss or sigmoid
        #T cons = 0 # for sigmoid

        # set and get
        kernel_type GetKernelType()
        void SetKernelType(kernel_type kt)

        # inner products
        @staticmethod
        inline T innerProduct(const TP* x, const TP* y, size_t d)
        @staticmethod
        inline void innerProducts(const TP* X, const TP* Y, size_t d, T* K, size_t m, size_t n)

        # squared distances
        @staticmethod
        inline T squaredDistance(const TP* x, const TP* y, size_t d)
        @staticmethod
        inline void squaredDistances(const TP* X, const TP* Y, size_t d, 
                T* K, size_t m, size_t n)


        # operators
        inline T operator() (const void* param, const TP* x, const TP* y, size_t d) const
        inline void operator() (const void* param, const TP* X, 
                const TP* Y, size_t d,T* K, size_t m, size_t n) const

        T (*user_element_function)(const void* param, const TP* y, size_t d)
        T (*user_matrix_function)(const void* param, const T* X, const T* Y, T*K, size_t m, size_t n)

    cdef cppclass KernelMatrix[T]:
        # symmetric constructor
        KernelMatrix( size_t m_, size_t n_, size_t d_, kernel_s[T,T] &kernel_,
                Data[T] & sources_)

        # non-symmetric constructor
        KernelMatrix( size_t m_, size_t n_, size_t d_, kernel_s[T,T] &kernel_,
                Data[T] & sources_, Data[T] & targets_)

        # operator
        T operator()(size_t i, size_t j)

        # return dimension
        size_t dim()

# tree.hpp import Tree
cdef extern from "${CMAKE_SOURCE_DIR}/gofmm/tree.hpp" namespace "hmlp::tree":
    cdef cppclass Tree[SETUP,NODEDATA]:
        pass



## gofmm.hpp import compress essentials 
cdef extern from "${CMAKE_SOURCE_DIR}/gofmm/gofmm.hpp" namespace "hmlp::gofmm":
    cdef cppclass centersplit[SPDMATRIX, int, T]:
        SPDMATRIX *Kptr

    cdef cppclass randomsplit[SPDMATRIX,int,T]:
        SPDMATRIX *Kptr

    # setup 
    cdef cppclass Setup[SPDMATRIX,SPLITTER,T]:
        pass

    # nodedata
    cdef cppclass NodeData[T]:
        pass

    #Compress (Turn into H Matrix)
    cdef Tree[Setup[ SPDMATRIX, centersplit[SPDMATRIX,two,T], T ], NodeData[T] ] *Compress[T,SPDMATRIX]( SPDMATRIX&, 
            T, T, size_t, size_t, size_t, bool)
    
    cdef Tree[Setup[ SPDMATRIX, centersplit[SPDMATRIX,two,T], T ], NodeData[T] ] *Compress[CSPLIT, RSPLIT, T,SPDMATRIX]( SPDMATRIX&, Data[pair[T, size_t]]&,
            centersplit[SPDMATRIX, two, T], randomsplit[SPDMATRIX, two, T], Configuration[T]&)

    #Evaluate (Matvec)
    cdef Data[T] Evaluate[use_runtime, use_omp_task, nnprune, cache, TREE, T](TREE &tr, Data[T] &weights)
    
    cpdef cppclass dTree_t:
        pass
    cpdef cppclass sTree_t:
        pass

    #Find Nearest Neighbors
    cdef Data[pair[T, size_t]] FindNeighbors[SPLITTER, T, SPDMATRIX](SPDMATRIX&, SPLITTER, Configuration[T]&, size_t)
    
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
    
    #Legacy Working prototype *Compress
    #sTree_t *Compress(sSPDMatrix_t*, float, float)
    #dTree_t *Compress(dSPDMatrix_t&, double, double)
    #dTree_t *Compress(SPDMatrix[double]&, double, double)
    #sTree_t *Compress(SPDMatrix[float]&, float, float)  
    
cdef extern from "${CMAKE_SOURCE_DIR}/gofmm/igofmm.hpp" namespace "hmlp::gofmm":
    cdef hmlpError_t Factorize[T,TREE](TREE , T )

    cdef hmlpError_t Solve[T,TREE](TREE, Data[T])
    #cdef hmlpError_t Solve[T,TREE](TREE, Data[T],Data[T])

