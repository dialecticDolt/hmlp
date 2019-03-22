#distutils: language = c++
from libcpp cimport bool as bool_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
cimport  mpi4py.libmpi as libmpi

from Config cimport *
from DistData cimport *
from DistMatrix cimport *

cdef extern from *:
    ctypedef int two "2"
    ctypedef bool use_runtime "true"
    ctypedef bool use_opm_task "true"
    ctypedef bool nnprune "true"
    ctypedef bool cache "true"

# used for tree templating later (no distributed nodedata)
cdef extern from "${CMAKE_SOURCE_DIR}/gofmm/gofmm.hpp" namespace "hmlp::gofmm" nogil:
    cdef cppclass NodeData[T]:
        pass

cdef extern from "${CMAKE_SOURCE_DIR}/gofmm/tree_mpi.hpp" namespace "hmlp::mpitree" nogil:
    cdef cppclass Tree[SETUP, NODEDATA]:
        Tree(libmpi.MPI_Comm) except +
        vector[size_t] getGIDS()

cdef extern from "${CMAKE_SOURCE_DIR}/gofmm/gofmm_mpi.hpp" namespace "hmlp::mpigofmm" nogil:
    #used to template Tree    
    cdef cppclass Setup[MATRIX, SPLITTER, T]:
        pass

    cdef cppclass centersplit[MATRIX, int, T]:
        MATRIX *Kptr

    cdef cppclass randomsplit[MATRIX, int, T]:
        MATRIX *Kptr

    cdef Tree[Setup[ SPDMATRIX, centersplit[SPDMATRIX,two,T], T ], NodeData[T] ]* Compress[CSPLIT, RSPLIT, T,SPDMATRIX]( SPDMATRIX&, STAR_CBLK_DistData[pair[T, size_t]]&, centersplit[SPDMATRIX, two, T], randomsplit[SPDMATRIX, two, T], Configuration[T]&,libmpi.MPI_Comm)

    #Evaluate (Matvec)

    RIDS_STAR_DistData[T]* Evaluate_Python_RIDS[NNPRUNE, TREE, T](TREE& tree, RIDS_STAR_DistData[T]& ddata) except +
    RBLK_STAR_DistData[T]* Evaluate_Python_RBLK[NNPRUNE, TREE, T](TREE&, RBLK_STAR_DistData[T]&) except +
    
    
    #TODO: 
    #       -Add FindNeighbors
    #       -Add ComputeError
    #       -LaunchHelper/SelfTesting ?



