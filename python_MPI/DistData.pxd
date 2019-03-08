#distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
cimport  mpi4py.libmpi as libmpi

cdef extern from "${CMAKE_SOURCE_DIR}/frame/base/DistData.hpp" namespace "hmlp":

    ctypedef enum Distribution_t "Distribution_t":
        CBLK "CBLK",
        RBLK "RBLK",
        CIDS "CIDS",
        RIDS "RIDS",
        USER "USER",
        STAR "STAR",
        CIRC "CIRC"

    ctypedef Distribution_t DTYPE "CIRC";    

    cdef cppclass DistData<DTYPE, DTYPE, T>[T]:
        DistData(size_t m, size_t n, int owner, libmpi.MPI_Comm comm) except +
        
        #TODO: Implement redistribute copy

        
        
