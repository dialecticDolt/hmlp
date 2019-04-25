from Runtime cimport *
from Config cimport *
from DistData cimport *
from DistMatrix cimport *
from Config cimport *
from DistTree cimport *
from DistTree cimport Compress as c_compress
from DistInverse cimport *

#Import MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi

#Import from cython: cpp
from libcpp.pair cimport pair
from libcpp cimport vector
from libcpp.map cimport map

cdef class PyRuntime:
    cdef int isInit

cdef class PyConfig:
    cdef Configuration[float]* c_config
    cpdef str metric_t

cdef class PyData:
    cdef Data[float]* c_data
    cdef deepcopy(self, PyData other)

cdef class PyDistData_CBLK:
    cdef STAR_CBLK_DistData[float]* c_data
    cdef MPI.Comm our_comm

cdef class PyDistData_RBLK:
    cdef RBLK_STAR_DistData[float]* c_data
    cdef MPI.Comm our_comm

cdef class PyDistData_RIDS:
    cdef RIDS_STAR_DistData[float]* c_data
    cdef MPI.Comm our_comm
    cdef map[size_t, size_t] rid2row
    cdef columndata(self, int i)
    cdef c_mult(self, RIDS_STAR_DistData[float] b)

cdef class PyDistPairData:
    cdef STAR_CBLK_DistData[pair[float, size_t]]* c_data
    
cdef class PyKernel:
    cdef kernel_s[float, float]* c_kernel
   
cdef class PyDistKernelMatrix:
    cdef DistKernelMatrix[float, float]* c_matrix


ctypedef Tree[Setup[DistKernelMatrix[float, float], centersplit[DistKernelMatrix[float, float], two , float], float], NodeData[float]] km_float_tree

cdef class PyTreeKM:
    cdef km_float_tree* c_tree
    cdef MPI.Comm our_comm
    cdef int cStatus
    cdef int fStatus
     
cdef class KernelMatrix:
    cdef PyKernel kernel
    cdef PyTreeKM tree
    cdef PyDistKernelMatrix K
    cdef PyConfig config
    cdef MPI.Comm comm_mpi
    cdef int is_compressed
    cdef int is_secacc
    cdef int is_factorized
    cdef int size


