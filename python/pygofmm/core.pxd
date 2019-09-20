from Runtime cimport *
from Config cimport *
from DistData cimport * 
from DistData cimport Data as c_Data
from DistMatrix cimport *
from DistMatrix cimport DistKernelMatrix as c_DistKernelMatrix
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

to_exclude = ['Data', 'DistKernelMatrix']
for name in to_exclude:
    del globals()[name]

cdef class Runtime:
    cdef int is_init

cdef class Config:
    cdef Configuration[float]* c_config
    cpdef str metric_t

cdef class LocalData:
    cdef c_Data[float]* c_data
    cdef deep_copy(self, LocalData other)

cdef class DistData_CBLK:
    cdef STAR_CBLK_DistData[float]* c_data
    cdef MPI.Comm our_comm

cdef class DistData_RBLK:
    cdef RBLK_STAR_DistData[float]* c_data
    cdef MPI.Comm our_comm

cdef class DistData_RIDS:
    cdef RIDS_STAR_DistData[float]* c_data
    cdef MPI.Comm our_comm
    cdef map[size_t, size_t] rid2row
    cdef columndata(self, int i)
    cdef c_mult(self, RIDS_STAR_DistData[float] b)

cdef class DistDataPair:
    cdef STAR_CBLK_DistData[pair[float, size_t]]* c_data
    
cdef class Kernel:
    cdef kernel_s[float, float]* c_kernel
   
cdef class Distributed_Kernel_Matrix:
    cdef c_DistKernelMatrix[float, float]* c_matrix


ctypedef Tree[Argument[c_DistKernelMatrix[float, float], centersplit[c_DistKernelMatrix[float, float], two , float], float], NodeData[float]] km_float_tree

cdef class KMTree:
    cdef km_float_tree* c_tree
    cdef MPI.Comm our_comm
    cdef int cStatus
    cdef int fStatus
     
cdef class KernelMatrix:
    cdef Kernel kernel
    cdef KMTree tree
    cdef Distributed_Kernel_Matrix K
    cdef Config config
    cdef MPI.Comm comm_mpi
    cdef int is_compressed
    cdef int is_secacc
    cdef int is_factorized
    cdef int size
    cdef int d
    cdef int nlocal

