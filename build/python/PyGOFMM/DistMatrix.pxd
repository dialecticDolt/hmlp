#distutils: language = c++
from libcpp cimport bool as bool_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
cimport  mpi4py.libmpi as libmpi

from DistData cimport *


cdef extern from "/workspace/will/dev/hmlp/frame/containers/KernelMatrix.hpp" namespace "hmlp" nogil:

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


    cdef cppclass kernel_s[T,TP]:
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

        T (*user_element_function)(const void* param, const TP* c, const TP* x, size_t d) nogil
        void (*user_matrix_function)(const void* param, const T* X, const T* Y, size_t d, T*K, size_t m, size_t n) nogil

    cdef cppclass DistKernelMatrix[T, TP, Allocator=*]:
        DistKernelMatrix(size_t, size_t, size_t, STAR_CBLK_DistData[T]&, STAR_CBLK_DistData[T]&, libmpi.MPI_Comm) except +
        DistKernelMatrix(size_t, size_t, STAR_CBLK_DistData[T]&, libmpi.MPI_Comm) except +
        
        DistKernelMatrix(size_t, size_t, size_t, kernel_s[T, TP]&, STAR_CBLK_DistData[T]&, STAR_CBLK_DistData[T]&, libmpi.MPI_Comm) except +
        DistKernelMatrix(size_t, size_t, kernel_s[T, TP]&, STAR_CBLK_DistData[T]&, libmpi.MPI_Comm) except +
    
        Data[T]& GeometryDistances(const vector[size_t]&, const vector[size_t]&) except +
        Data[T]& Diagonal(vector[size_t]&) except +
        pair[T, size_t] ImportantSample(size_t) except +
        void Print()
        void SendIndices(vector[size_t]&, int, libmpi.MPI_Comm)
        void RecvIndices(int, libmpi.MPI_Comm, libmpi.MPI_Status)
        void BcasIndices(vector[size_t]&, int, libmpi.MPI_Comm)
        void RequestIndices(const vector[vector[size_t]]&)
        size_t dim()
        T operator () (size_t i, size_t j)         
