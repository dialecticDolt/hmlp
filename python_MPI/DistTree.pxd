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



cdef extern from "${CMAKE_SOURE_DIR}/gofmm/tree_mpi.hpp" namespace "hmlp:mpitree":

    cdef cppclass Tree[SETUP, NODEDATA]:
        pass

cdef extern from "${CMAKE_SOURCE_DIR}/gofmm/gofmm_mpi.hpp" namespace "hmlp::gofmm":

    #Objects used to template Tree    
    cdef cppclass centersplit[MATRIX, int, T]:
        MATRIX *Kptr

    cdef cppclass randomsplit[MATRIX, int, T]:
        MATRIX *Kptr

    cdef cppclass Setup[MATRIX, SPLITTER, T]:
        pass

    cdef cppclass NodeData[T]:
        pass


    #Compress

    #Float (single precision)
    #cdef Tree[Setup[MATRIX, centersplit[MATRIX, two, float], float], NodeData[float]] *Compress[CSPLIT, RSPLIT, float, MATRIX](MATRIX&, STAR_CBLK_DistData[pair[float, size_t]]&, centersplit[MATRIX, two, float], randomsplit[MATRIX, two, float], Configuration[float]&)

    cdef Tree[Setup[ SPDMATRIX, centersplit[SPDMATRIX,two,T], T ], NodeData[T] ] *Compress[CSPLIT, RSPLIT, T,SPDMATRIX]( SPDMATRIX&, STAR_CBLK_DistData[pair[T, size_t]]&,
            centersplit[SPDMATRIX, two, T], randomsplit[SPDMATRIX, two, T], Configuration[T]&)

    #Evaluate (Matvec)

    #Float (single precision)
    #RIDS_STAR_f_DistData[float] Evaluate[NNPRUNE, TREE, float](TREE&, RIDS_STAR_f_DistData[float]&)
    #RBLK_STAR_f_DistData[float] Evaluate[NNPRUNE, TREE, float](TREE&, RBLK_STAR_f_DistData[float]&)
    

    #TODO: 
    #       -Add FindNeighbors
    #       -Add ComputeError
    #       -LaunchHelper/SelfTesting ?



