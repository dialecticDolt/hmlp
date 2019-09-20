#Import our headers and functions from GOFMM
from DistData cimport *
from Runtime cimport *
from DistMatrix cimport *
from DistMatrix cimport DistKernelMatrix as c_DistKernelMatrix
from CustomKernel cimport *
from Config cimport *
from DistTree cimport *
from DistTree cimport Compress as c_compress
from DistInverse cimport *

#Import MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi
from mpi4py import MPI

#Import from cython: cpp
from libcpp.pair cimport pair
from libcpp cimport vector
from libcpp.map cimport map

#Import from cython: c
from libc.math cimport sqrt
from libc.string cimport strcmp
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

#Import from cython: misc
import cython
from cython.operator cimport dereference as deref
from cython.view cimport array as cvarray
from cython.parallel cimport prange

#Import numpy
import numpy as np
cimport numpy as np
np.import_array()

"""@package docstring
Documentation for PyGOFMM.Core

More details here.

"""

def reformat(numpyArray, flatten=False):
    """Reformat a Numpy array into Float32 datatype with fortran (column-major) ordering.
       This allows it to be passed to HMLP internals. 
       Note: that if it is not in this format a copy will be performed. 
    """
    if flatten:
        return np.asarray(numpyArray, dtype='float32', order='F').flatten()
    else:        
        return np.asarray(numpyArray, dtype='float32', order='F')

def get_cblk_ownership(int N, int rank, int nprocs):
    """Return the indicies (GIDS) for N points owned by n processors in block cyclic ordering"""
    #Note: this copies the data
    cblk_idx = np.asarray(np.arange(rank, N, nprocs), dtype='int32', order='F')
    return cblk_idx

def distribute_cblk(MPI.Comm comm, points):
    """ Distribute the given array over the communicator in column block cyclic ordering"

        Arguments: 
        comm - MPI.Comm communicator object from mpi4py
        points - d x N numpy array. Each MPI process with recieve a d x ~N/n block.

        Output:
        Tuple of (block, index)
        block - DistData_CBLK containing d x ~N/n points
        index - the index (GID) set of points in sources
    """
    cdef int N, d, nprocs, rank
    N = np.size(points[0, :])
    d = np.size(points[:, 0])
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    
    index = get_cblk_ownership(N, rank, nprocs)
    points = reformat(points)
    pointsCBLK = points[:, index]
    CBLK_points = reformat(pointsCBLK)
    block = DistData_CBLK(comm, d, N, darr=CBLK_points)
    return (block, index)

cdef class Runtime:
    """
    Wrapper class for HMLP Runtime system. (Coordiates tasks)
    Usage: 
        Call init before use of PyGOFMM functions
        Call finalize at the end the section that uses PyGOFMM functionality
        (Note: finalize is automatically called when a runtime object is deallocated)
    """
    def __cinit__(self):
        self.is_init = int(0)

    def __dealloc__(self):
        hmlp_finalize()

    def initialize(self, MPI.Comm comm=None):
        """ Starts the runtime system. 
            Keyword Arguments:
            comm - an MPI.Comm communicator object from mpi4py
        """
        cdef int arg_c=7
        cdef char **arg_v = <char **>malloc(7* sizeof(char*))
        if(comm):
            hmlp_init(comm.ob_mpi)
        else:
            hmlp_init(&arg_c, &arg_v)

    def init_with_MPI(self, MPI.Comm comm):
        hmlp_init(comm.ob_mpi)

    def set_workers(self, int nworkers):
        if self.isInit is 1:
            hmlp_set_num_workers(nworkers)

    def run(self):
        hmlp_run()

    def finalize(self):
        """ Cleans up the runtime system. """
        hmlp_finalize()


def convert_metric(str metric_type):
    """ Converts string of metric name to enum value used by GOFMM """
    if(metric_type == "GEOMETRY_DISTANCE"):
        m = int(0)
    elif(metric_type == "KERNEL_DISTANCE"):
        m = int(1)
    elif(metric_type == "ANGLE_DISTANCE"):
        m = int(2)
    elif(metric_type == "USER_DISTANCE"):
        m = int(3)
    return m


cdef class Config:
    """ Wrapper class for GOFMM Configuration Object.
        Stores parameters related to kernel compression

        metric_type
        problem_size
        leaf_node_size
        neighbor_size
        maximum_rank
        tolerance
        budget
    """
    def __cinit__(self, str metric_type="GEOMETRY_DISTANCE", int problem_size=2000, int leaf_node_size=128, int neighbor_size=64, int maximum_rank=128, float tolerance=0.0001, float budget=0.01, bool secure_accuracy=True):
        self.metric_t = metric_type
        m = convert_metric(metric_type) 
        m = int(m)
        self.c_config = new Configuration[float](m, problem_size, leaf_node_size, neighbor_size, maximum_rank, tolerance, budget, secure_accuracy)

    def set_all(self, str metric_type, int problem_size, int leaf_node_size, int neighbor_size, int maximum_rank, float tolerance, float budget, bool secure_accuracy):
        self.metric_t = metric_type
        m = convert_metric(metric_type)
        m = int(m)
        self.c_config.Set(m, problem_size, leaf_node_size, neighbor_size, maximum_rank, tolerance, budget, secure_accuracy)

    #TODO: Add getters and setters for all
    def set_metric_type(self, metric_type):
        self.metric_t = metric_type
        m = convert_metric(metric_type)
        m = int(m)
        self.c_config.Set(m, self.getProblemSize(), self.getLeafNodeSizei(), self.getNeighborSize(), self.getMaximumRank(), self.getTolerance(), self.getBudget(), self.isSecure())

    def set_neighbor_size(self, int nsize):
        self.c_config.Set(self.c_config.MetricType(), self.getProblemSize(), self.getLeafNodeSize(), nsize, self.getMaximumRank(), self.getTolerance(), self.getBudget(), self.isSecure())

    def set_problem_size(self, int psize):
        self.c_config.Set(self.c_config.MetricType(), psize, self.getLeafNodeSizei(), self.getNeighborSize(), self.getMaximumRank(), self.getTolerance(), self.getBudget(), self.isSecure())

    def set_maximum_rank(self, int mrank):
        self.c_config.Set(self.c_config.MetricType(), self.getProblemSize(), self.getLeafNodeSize(), self.getNeighborSize(), mrank, self.getTolerance(), self.getBudget(), self.isSecure())

    def set_tolerance(self, float tol):
        self.c_config.Set(self.c_config.MetricType(), self.getProblemSize(), self.getLeafNodeSize(), self.getNeighborSize(), self.getMaximumRank(), tol, self.getBudget(), self.isSecure())

    def set_budget(self, float budget):
        self.c_config.Set(self.c_config.MetricType(), self.getProblemSize(), self.getLeafNodeSize(), self.getNeighborSize(), self.getMaximumRank(), self.getTolerance(), budget, self.isSecure())

    def set_secure_accuracy(self, bool status):
        self.c_config.Set(self.c_config.MetricType(), self.getProblemSize(), self.getLeafNodeSize(), self.getNeighborSize(), self.getMaximumRank(), self.getTolerance(), self.getBudget(), status)
        
    def get_metric_type(self):
        return self.metric_t

    def get_maximum_rank(self):
        return self.c_config.MaximumRank()

    def get_neighbor_size(self):
        return self.c_config.NeighborSize()

    def get_problem_size(self):
        return self.c_config.ProblemSize()

    def get_maximum_depth(self):
        return self.c_config.getMaximumDepth()

    def get_leaf_node_size(self):
        return self.c_config.getLeafNodeSize()

    def get_tolerance(self):
        return self.c_config.Tolerance()

    def get_budget(self):
        return self.c_config.Budget()

    def is_symmetric(self):
        return self.c_config.IsSymmetric()

    def is_adaptive(self):
        return self.c_config.UseAdaptiveRanks()

    def is_secure(self):
        return self.c_config.SecureAccuracy()

    def set_leaf_node_size(self, int leaf_node_size):
        self.c_config.setLeafNodeSize(leaf_node_size)

    def set_adaptive_rank(self, bool status):
        self.c_config.setAdaptiveRanks(status)

    def set_symmetry(self, bool status):
        self.c_config.setSymmetric(status)



cdef class LocalData:
    """ Wrapper Class for HMLP's local Data object. """

    #TODO: Add error handling if the user gives the wrong array sizes (at the moment I'm overriding)
    @cython.boundscheck(False)
    def __cinit__(self, size_t m=0, size_t n=0, str filename=None, float[::1, :] darr=None, float[:] arr=None, LocalData data=None):
        cdef string c_filename
        cdef vector[float] vec
        cdef int vec_sz
        if filename and not (data or darr!=None or arr!=None):
            #Load data object from file
            c_filename = <string>filename.encode('utf-8')
            with nogil:
                self.c_data = new c_Data[float](m, n)
                self.c_data.readBinaryFile(m, n, c_filename)
        elif data and not (filename or darr!=None or arr!=None):
            #Deep copy from existing data object
            with nogil:
                self.c_data = new c_Data[float](deref(data.c_data))
        elif darr!=None and not (filename or data!=None or arr!=None):
            #Load data object from 2D numpy array
            m = <size_t>darr.shape[0]
            n = <size_t>darr.shape[1]
            vec_sz = <int> (m * n)
            with nogil:
                vec.assign(&darr[0, 0], &darr[0,0] + vec_sz)
                self.c_data = new c_Data[float](m, n, &darr[0, 0], True)
        elif arr!=None and not (filename or data!=None or darr!=None):
            #Load data object from 1D numpy array
            m = <size_t>arr.size
            n = <size_t>1
            vec_sz = len(arr)
            with nogil:
                vec.assign(&arr[0], &arr[0] + vec_sz)
                self.c_data = new c_Data[float](m, n, vec)
        else:
            #Create empty Data object
            with nogil:
                self.c_data = new c_Data[float](m, n)

    def __dealloc__( self ):
        print("Cython: Running __dealloc___ for LocalData Object")
        # self.c_data.clear()
        free(self.c_data)

    def __getitem__(self, pos):
        if isinstance(pos, int):
            return self.c_data.getvalue(<size_t>pos,<size_t>0)
        elif isinstance(pos, tuple) and len(pos) == 2:
            i, j = pos
            return self.c_data.getvalue(i, j)
        else:
            raise Exception('LocalData can only be indexed in 1 or 2 dimensions')

    def __setitem__(self, pos, float v):
        if isinstance(pos, int):
            self.c_data.setvalue(<size_t>pos,<size_t>0, v)
        elif not isinstance(pos, int) and len(pos) == 2:
            i, j = pos
            self.c_data.setvalue(<size_t>i, <size_t>j, v)
        else:
            raise Exception('LocalData can only be indexed in 1 or 2 dimensions')

    @cython.boundscheck(False)
    @classmethod 
    def from_numpy(cls,np.ndarray[float, ndim=2, mode="c"] arr_np):
        # get sizes
        cdef size_t m,n
        m = <size_t> arr_np.shape[0]
        n = <size_t> arr_np.shape[1]
        # construct LocalData obj
        cpdef LocalData ret = cls(m,n)
        cdef c_Data[float]* bla 
        with nogil:
            bla = new c_Data[float](m,n,&arr_np[0,0],True)
        ret.c_data = bla
        return ret

    @cython.boundscheck(False)
    def get_submatrix(self,np.ndarray[np.intp_t, ndim=1] I not None,
        np.ndarray[np.intp_t,ndim=1] J not None):

        # define memory views
        cdef np.intp_t [:] Iview = I
        cdef np.intp_t [:] Jview = J

        cdef size_t ci,cj

        # get sizes, initialize new PyData
        cdef size_t ni = <size_t> I.size
        cdef size_t nj = <size_t> J.size
        cdef c_Data[float]* subdata = new c_Data[float](ni,nj)
        cdef float tmp

        # begin loops
        for ci in range(ni):
            for cj in range(nj):
                tmp = self.c_data.getvalue( <size_t> Iview[ci], <size_t> Jview[cj] )
                subdata.setvalue(<size_t> ci,<size_t> cj,tmp)

        # new Pydata object
        cpdef LocalData sub = LocalData(ni,nj)
        
        # call c_data's sub func
        sub.c_data = subdata

        # return sub
        return sub

    def read(self, size_t m, size_t n, str filename):
        cdef string c_filename
        c_filename = <string>filename.encode('utf-8')
        with nogil:
            self.c_data.readBinaryFile(m, n,c_filename)

    def write(self,str filename):
        self.c_data.writeBinaryFile(filename.encode())

    def rows(self):
        return self.c_data[0].row()

    def cols(self):
        return self.c_data.col()

    def size(self):
        return self.c_data.size()

    def set_value(self,size_t m, size_t n, float v):
        self.c_data.setvalue(m,n,v)
    
    def get_value(self, size_t m, size_t n):
        return self.c_data.getvalue(m, n)

    def rand(self,float a, float b ):
        with nogil:
            self.c_data.rand(a, b)

    def randn(self, float mu, float std):
        with nogil:
            self.c_data.randn(mu, std)

    def randspd(self, float a, float b):
        with nogil:
            self.c_data.randspd(a, b)

    def display(self):
        self.c_data.Print()

    def has_illegal_value(self):
        return self.c_data.HasIllegalValue()

    def make_copy(self):
        # get my data stuff
        cdef c_Data[float]* cpy = new c_Data[float](deref(self.c_data) )
        # put into python obj
        cpdef LocalData bla = LocalData(self.row(), self.col())
        bla.c_data = cpy
        return bla

    cdef deep_copy(self,LocalData other):
        del self.c_data
        self.c_data = new c_Data[float]( deref(other.c_data) )

    def to_numpy(self):
        cdef float* data_ptr = self.c_data.rowdata(0)
        cdef float[:] mv = <float[:self.size()]> data_ptr
        np_arr = np.asarray(mv, order='F', dtype='float32').reshape((self.rows(),self.cols()),order='F')
        return np_arr

    def to_array(self, copy=False, flatten=False):
        cdef float* data_ptr = self.c_data.rowdata(0)
        cdef float[:] mv = <float[:self.size()]> data_ptr
        np_arr = np.asarray(mv, order='F', dtype='float32')
        if not flatten:
            np_arr = np_arr.reshape((self.rows(), self.cols()), order='F')
        if copy:
            return np.copy(np_arr)
        else:
            return np_arr


#Python Class for Distributed Data Object - Columnwise Block Cyclic
cdef class DistData_CBLK:

    #TODO: Fix loading from local PyData objects
    @cython.boundscheck(False)
    def __cinit__(self, MPI.Comm comm, size_t m=0, size_t n=0, str filename=None, float[::1, :] darr=None, float[:] arr=None, LocalData data=None):
        cdef string c_filename
        cdef vector[float] vec
        cdef int vec_sz
        if filename and not (data or darr!=None or arr!=None):
            #Load data object from file
            c_filename = <string>filename.encode('utf-8')
            with nogil:
                self.c_data = new STAR_CBLK_DistData[float](m, n, comm.ob_mpi, c_filename)
        elif data and not (filename or darr!=None or arr!=None):
            #From local copy of LocalData object TODO TODO TODO Same error as dist data deep copy constructors (arguments don't match)
            raise Exception("Error: Cannot copy from local Data object to DistData object. This feature is currently broken")
            self.c_data = new STAR_CBLK_DistData[float](m, n, deref(<Data[float]*>(LocalData(data).c_data)), comm.ob_mpi)
        elif darr!=None and not (filename or data!=None or arr!=None):
            #Load data object from 2d numpy array
            vec_sz = darr.shape[0] * darr.shape[1]
            with nogil:
                vec.assign(&darr[0, 0], &darr[0,0] + vec_sz)
                self.c_data = new STAR_CBLK_DistData[float](m, n, vec, comm.ob_mpi)
        elif arr!=None and not (filename or data!=None or darr!=None):
            #Load data object from numpy array
            vec.assign(&arr[0], &arr[0] + len(arr))
            with nogil:
                self.c_data = new STAR_CBLK_DistData[float](m, n, vec, comm.ob_mpi)
        else:
            #Create empty Data object
            with nogil:
                self.c_data = new STAR_CBLK_DistData[float](m, n, comm.ob_mpi)

    def __len__(self):
        return self.c_data.row()*self.c_data.col()

    def __getitem__(self, pos):
        if isinstance(pos, int) and self.c_data.col() == 1:
            return deref(self.c_data)(<size_t>pos,<size_t>1)
        #elif isinstance(pos, int) and self.c_data.row() == 1:
        #    return deref(self.c_data)(<size_t>1, <size_t>pos)
        elif len(pos) == 2:
            i, j = pos
            return deref(self.c_data)(<size_t>i, <size_t>j)
        else:
            raise Exception('DistData can only be indexed in 1 or 2 dimensions')

    def __dealloc__(self):
        print("Cython: Running __dealloc___ for DistData Object")
        free(self.c_data)
    
    def rand(self, float a=0.0, float b=1.0):
        with nogil:
            self.c_data.rand(a, b)

    def display(self):
        self.c_data.Print()

    def randn(self, float m=0.0, float s=1.0):
        with nogil:
            self.c_data.randn(m, s)

    def get_comm_size(self):
        return self.c_data.GetSize()

    def get_mpi_rank(self):
        return self.c_data.GetRank()

    def size(self):
        return self.rows() * self.cols()

    def rows(self):
        return self.c_data.row()

    def cols(self):
        return self.c_data.col()

    def local_rows(self):
        return self.c_data.row_owned()

    def local_cols(self):
        return self.c_data.col_owned()

    def nlocal(self):
        return self.c_data.col_owned()

    def to_array(self, copy=False, flatten=False):
        cdef float* data_ptr = self.c_data.rowdata(0)
        cdef float[:] mv = <float[:self.size()]> data_ptr
        np_arr = np.asarray(mv, order='F', dtype='float32')
        if not flatten:
            np_arr = np_arr.reshape((self.local_rows(), self.local_cols()), order='F')
        if copy:
            return np.copy(np_arr)
        else:
            return np_arr


cdef class DistData_RBLK:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, MPI.Comm comm, size_t m=0, size_t n=0, str filename=None, float[::1, :] darr=None, float[:] arr=None, LocalData data=None):
        cdef string c_filename
        cdef vector[float] vec
        cdef int vec_sz

        if filename and not (data or darr!=None or arr!=None):
            #Load data object from file
            raise Exception("RBLK does not support loading from a file. Please load a RIDS and convert if you need a RBLK")
        elif data and not (filename or darr!=None or arr!=None):
            #From local copy of PyData object
            raise Exception("Error: Cannot copy from local Data object to DistData object. This feature is currently broken")
            self.c_data = new RBLK_STAR_DistData[float](m, n, deref(<Data[float]*>(LocalData(data).c_data)), comm.ob_mpi)
        elif darr!=None and not (filename or data!=None or arr!=None):
            #Load data object from 2D numpy array
            vec_sz = darr.shape[0] * darr.shape[1]
            with nogil:
                vec.assign(&darr[0, 0], &darr[0,0] + vec_sz)
                self.c_data = new RBLK_STAR_DistData[float](m, n, vec, comm.ob_mpi)
        elif arr!=None and not (filename or data!=None or darr!=None):
            #Load data object from numpy array
            vec_sz = len(arr)
            with nogil:
                vec.assign(&arr[0], &arr[0] + vec_sz)
                self.c_data = new RBLK_STAR_DistData[float](m, n, vec, comm.ob_mpi)
        else:
            #Create empty Data object
            with nogil:
                self.c_data = new RBLK_STAR_DistData[float](m, n, comm.ob_mpi)

    def __dealloc__(self):
        print("Cython: Running __dealloc___ for DistData Object")
        free(self.c_data)

    #TODO: Add a class method for this
    def load_from_rids(self, DistData_RIDS b):
        with nogil:
            self.c_data[0] = (deref(b.c_data))

    def __getitem__(self, pos):
        if isinstance(pos, int) and self.c_data.col() == 1:
            return deref(self.c_data)(<size_t>pos,<size_t>0)
        elif len(pos) == 2:
            i, j = pos
            return deref(self.c_data)(<size_t>i, <size_t>j)
        else:
            raise Exception('DistData can only be indexed in 1 or 2 dimensions')

    def rand(self, float a=0.0, float b=1.0):
        with nogil:
            self.c_data.rand(a, b)

    def display(self):
        self.c_data.Print()

    def randn(self, float m=0.0, float s=1.0):
        with nogil:
            self.c_data.randn(m, s)

    def get_comm_size(self):
        return self.c_data.GetSize()

    def get_rank(self):
        return self.c_data.GetRank()

    def rows(self):
        return self.c_data.row()

    def cols(self):
        return self.c_data.col()

    def local_rows(self):
        return self.c_data.row_owned()

    def local_cols(self):
        return self.c_data.col_owned()

    def nlocal(self):
        return self.c_data.row_owned()


cdef class DistData_RIDS:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, MPI.Comm comm,size_t m=0, size_t n=0, size_t mper = 0, str filename=None, int[:] iset=None, \
            float[::1,:] darr=None, float[::1] arr=None, KMTree tree=None, LocalData data=None, DistData_RIDS ddata=None):
        cdef string c_filename
        cdef int rank
        rank = comm.Get_rank()

        if (arr!=None):
            mper = len(arr)
        if (darr!=None):
            mper = darr.shape[0]
        
        cdef int[:] a = np.arange(rank*mper, rank*mper+mper).astype('int32')
        cdef vector[size_t] own
        cdef vector[float] dat
        cdef int vec_sz

        # assign owned indices (own)
        if tree:
            own = tree.c_tree.getGIDS()
        elif iset==None and tree==None:
            own.assign(&a[0], &a[0] + len(a))
        else:
            own.assign(&iset[0], &iset[0] + len(iset))
            
        # call DistData constructor 
        if filename and not (data or darr!=None or arr!=None):
            #Load data object from file
            c_filename = <string>filename.encode('utf-8')
            with nogil:
                self.c_data = new RIDS_STAR_DistData[float](m, n, c_filename, comm.ob_mpi)
        elif data and not (filename or darr!=None):
            #From local copy of PyData object
            raise Exception("Error: Cannot create DistData from local Data objects. This feature is currently broken.")
        elif arr!=None and not (filename or darr!=None):
            #Load data object from 1D numpy array
            vec_sz = len(arr)
            with nogil:
                dat.assign(&arr[0], &arr[0] + vec_sz)
                self.c_data = new RIDS_STAR_DistData[float](m,n,own,dat,comm.ob_mpi)
        elif darr!=None and not (filename or data!=None):
            #Load data object from 2D numpy array
            vec_sz = darr.shape[0] * darr.shape[1] 
            with nogil:
                dat.assign(&darr[0,0], &darr[0,0] + vec_sz)
                self.c_data = new RIDS_STAR_DistData[float](m,n,own,dat,comm.ob_mpi)
        elif ddata and not (filename or darr!=None or data):
            #Copy from existing DistData_RIDS object
            with nogil:
                self.c_data = new RIDS_STAR_DistData[float](deref(ddata.c_data), comm.ob_mpi)
        else:
            #Create empty DistData_RIDS
            with nogil:
                self.c_data = new RIDS_STAR_DistData[float](m, n, own, comm.ob_mpi)

        self.rid2row = self.c_data.getMap()

    def __dealloc__(self):
        print("Cython: Running __dealloc___ for DistData Object")
        free(self.c_data)

    def rand(self, float a=0.0, float b=1.0):
        self.c_data.rand(a, b)

    def display(self):
        self.c_data.Print()

    def randn(self, float m=0.0, float s=1.0):
        self.c_data.randn(m, s)
   
    def load_from_rblk(self, DistData_RBLK b):
        with nogil:
            self.c_data[0] = (deref(b.c_data))

    def redistribute(self, DistData_RIDS b):
        with nogil:
            self.c_data[0] = (deref(b.c_data))

    def to_numpy(self):
        cdef float* data_ptr = self.c_data.rowdata(0)
        cdef float[:] mv = <float[:self.size()]> data_ptr
        np_arr = np.asarray(mv)
        np_arr.resize(self.local_rows(),self.local_cols())
        return np_arr

    def to_array(self, copy=False, flatten=False):
        cdef float* local_data
        local_data = self.c_data.rowdata(0)
        cdef float[:] mv = <float[:self.c_data.size()]> local_data
        np_arr = np.asarray(mv, order='F', dtype='float32')
        if not flatten:
            np_arr = np_arr.reshape((self.local_rows(), self.local_cols()), order='F')
        if copy:
            return np.copy(np_arr)
        else:
            return np_arr

    def __setitem__(self, pos, float v):
        i, j = pos
        i = self.rid2row[i]
        self.c_data.setvalue(<size_t>i, <size_t>j, v)

    @cython.boundscheck(False)
    @cython.boundscheck(False)
    def update_rids(self,int[:] iset):
        cdef vector[size_t] vec
        cdef int vec_sz
        vec_sz = len(iset)
        with nogil:
            vec.assign(&iset[0],&iset[0] + vec_sz)
            self.c_data.UpdateRIDS(vec)
            self.rid2row = self.c_data.getMap()

    def mult(self, DistData_RIDS b):
        return self.c_mult(deref(b.c_data))

    #Compute Self'*B
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cdef c_mult(self, RIDS_STAR_DistData[float] b):
        cdef float* b_data
        cdef float* a_data
        cdef int m, n, i, l, j, r
        b_data = b.columndata(0)
        a_data = self.c_data.columndata(0)
        m = self.cols()
        n = self.rows()
        assert(n == b.row())
        l = b.col()
        cdef float[:, :] C = np.zeros([m, l], dtype='float32', order='F')
        print(a_data[1])
        a_data[1] = 100
        print(a_data[1])
        for i in xrange(m):
            for j in xrange(l):
                for r in prange(n, nogil=True):
                    C[i, j] = 1
        
        return C

    cdef columndata(self, int i):
        cdef float * d
        d = deref(self.c_data).columndata(i)
        #return d

    def get_mpi_rank(self):
        return self.c_data.GetRank()

    def rows(self):
        return self.c_data.row()

    def cols(self):
        return self.c_data.col()

    def __getitem__(self, pos):
        if isinstance(pos, int) and self.c_data.col() == 1:
            return deref(self.c_data)(<size_t>pos,<size_t>0)
        elif len(pos) == 2:
            i, j = pos
            return deref(self.c_data)(<size_t>i, <size_t>j)
        else:
            raise Exception('DistData can only be indexed in 1 or 2 dimensions')

    def get_rids(self):
        cdef vector[size_t] rids_vec
        rids_vec = self.c_data.getRIDS()
        return np.asarray(rids_vec)

    def local_rows(self):
        return self.c_data.row_owned()

    def local_cols(self):
        return self.c_data.col_owned()

    def nlocal(self):
        return self.c_data.row_owned()

cdef class DistDataPair:

    #TODO: Loading from file, how to handle tuples?
    def __cinit__(self, MPI.Comm comm, size_t m, size_t n, str filename=None, localdata=None):
        cdef string c_filename
        if not (filename or localdata): 
            #create empty DistDataPair object
            with nogil:
                self.c_data = new STAR_CBLK_DistData[pair[float, size_t]](m, n, comm.ob_mpi)
        elif filename and not (localdata):
            #Load DistDataPair from file
            c_filename = <string>filename.encode('utf-8')
            with nogil:
                self.c_data = new STAR_CBLK_DistData[pair[float, size_t]](m, n, comm.ob_mpi, c_filename)
        elif localdata and not (filename):
            if type(localdata) is LocalData:
                self.c_data = new STAR_CBLK_DistData[pair[float, size_t]](m, n, deref(<Data[pair[float, size_t]]*>(LocalData(localdata).c_data)), comm.ob_mpi)
            if isinstance(localdata, (np.ndarray, np.generic)):
                raise Exception("Creating DistDataPair with local numpy arrays is not supported")
        else:
            raise Exception("DistDataPair: Invalid Constructor Parameters")

    def __dealloc__(self):
        print("Cython: Running __dealloc___ for DistData Object")
        self.c_data.clear()
        free(self.c_data)
    
    def __getitem__(self, pos):
        if isinstance(pos, int) and self.c_data.col() == 1:
            return deref(self.c_data)(<size_t>pos,<size_t>1)
        elif len(pos) == 2:
            i, j = pos
            return deref(self.c_data)(<size_t>i, <size_t>j)
        else:
            raise Exception('DistData can only be indexed in 1 or 2 dimensions')

    def to_numpy(self):
        cdef int local_cols
        cdef int local_rows
        local_cols = self.local_cols()
        local_rows = self.local_rows()
        
        cdef float[:, :] mv_distances = np.empty((local_rows, local_cols), dtype='float32')
        cdef size_t[:, :] mv_gids = np.empty((local_rows, local_cols), dtype='uintp')
        
        #Unpack pair[distance, gid]
        #TODO: Parallize this
        for i in range(local_rows):
            for j in range(local_cols):
                mv_distances[i, j] = self[i, j][0]
                mv_gids[i, j] = self[i, j][1]

        np_distances = np.asarray(mv_distances)
        np_gids = np.asarray(mv_gids)
        return (np_distances, np_gids)

    def get_comm_size(self):
        return self.c_data.GetSize()

    def get_rank(self):
        return self.c_data.GetRank()

    def rows(self):
        return self.c_data.row()

    def cols(self):
        return self.c_data.col()

    def local_rows(self):
        return self.c_data.row_owned()

    def local_cols(self):
        return self.c_data.col_owned()

cdef class Kernel:
    """ Python Class for Kernel Evaluation Object kernel_s """
    
    # constructor 
    def __cinit__(self, str kstring="GAUSSIAN"):
       self.c_kernel = new kernel_s[float,float]()
       k_enum = Kernel.get_kernel_type(kstring)
       self.c_kernel.SetKernelType(k_enum)
       if(k_enum==9):
            self.c_kernel.user_element_function = custom_element_kernel[float, float]
            self.c_kernel.user_matrix_function = custom_matrix_kernel[float, float]

    def __dealloc__(self):
        print("Cython: Running __dealloc__ for PyKernel")
        self.c_kernel.user_matrix_function = NULL
        self.c_kernel.user_element_function = NULL
        free(self.c_kernel)

    @staticmethod
    def get_kernel_type(str kstring):
        if(kstring == "GAUSSIAN"):
            m = int(0)
        elif(kstring == "SIGMOID"):
            m = int(1)
        elif(kstring == "POLYNOMIAL"):
            m = int(2)
        elif(kstring == "LAPLACE"):
            m = int(3)
        elif(kstring == "GAUSSIAN_VAR_BANDWIDTH"):
            m = int(4)
        elif(kstring == "TANH"):
            m = int(5)
        elif(kstring == "QUARTIC"):
            m = int(6)
        elif(kstring == "MULTIQUADRATIC"):
            m = int(7)
        elif(kstring == "EPANECHNIKOV"):
            m = int(8)
        elif(kstring == "USER_DEFINED"):
            m = int(9)
        else:
            raise ValueError("This is not a valid Kernel Type in PyGOFMM. Please use USER_DEFINED to specify custom kernels")
        return m

    # Gaussian param set/get
    def set_bandwidth(self,float _scal):
        self.c_kernel.scal = -0.5 / (_scal *_scal)

    def get_bandwidth(self):
        f = -0.5 / self.c_kernel.scal
        return sqrt(f)

    def set_scale(self,float _scal):
        self.c_kernel.scal = _scal

    def get_scale(self):
        return self.c_kernel.scal

    def set_custom_function(self, f, g):
        self.user_element_function = f
        self.user_matrix_function = g


cdef class Distributed_Kernel_Matrix:

    def __cinit__(self, MPI.Comm comm, Kernel kernel, DistData_CBLK sources, DistData_CBLK targets=None):
        cdef size_t m, d, n
        m = sources.c_data.col()
        d = sources.c_data.row()
        n = m

        if targets is not None:
            n = targets.col()
            with nogil:
                self.c_matrix = new c_DistKernelMatrix[float, float](m, n, d, deref(kernel.c_kernel), deref(sources.c_data), deref(targets.c_data), comm.ob_mpi)
        else:
            with nogil:
                self.c_matrix = new c_DistKernelMatrix[float, float](m, d, deref(kernel.c_kernel), deref(sources.c_data), comm.ob_mpi)

    def dim(self):
        return self.c_matrix.dim()

    def __getitem__(self, pos):
        if len(pos) == 2:
            i, j = pos
            return deref(self.c_matrix)(<size_t>i, <size_t>j)

    def get_value(self, i, j):
        return self[i, j]

cdef class KMTree:
    """ Class for Kernel Matrix Tree  """
    def __cinit__(self, MPI.Comm comm):
        with nogil:
            self.c_tree = new km_float_tree(comm.ob_mpi)
        self.our_comm = comm
        self.cStatus = 0
        self.fStatus = 0

    def __dealloc__(self):
        print("Cython: Running __dealloc__ for KMTree")
        free(self.c_tree)

    def get_gids(self):
        cdef vector[size_t] gidvec
        with nogil:
            gidvec = self.c_tree.getGIDS()
        np_arr = np.asarray(gidvec, dtype='int32')
        return np_arr

    def compress(self, MPI.Comm comm, Distributed_Kernel_Matrix K, float stol=0.001, float budget=0.01, size_t m=128, size_t k=64, size_t s=32, bool sec_acc=True, str metric_type="ANGLE_DISTANCE", bool sym=True, bool adapt_ranks=True, Config config=None):
        cdef centersplit[c_DistKernelMatrix[float, float], two, float] c_csplit
        cdef randomsplit[c_DistKernelMatrix[float, float], two, float] c_rsplit
        cdef STAR_CBLK_DistData[pair[float, size_t]]* c_NNdata = new STAR_CBLK_DistData[pair[float, size_t]](0, 0, comm.ob_mpi)
        c_csplit.Kptr = K.c_matrix
        c_rsplit.Kptr = K.c_matrix
        self.cStatus=1
        if(config):
            with nogil:
                self.c_tree = c_compress[centersplit[c_DistKernelMatrix[float, float], two, float], randomsplit[c_DistKernelMatrix[float, float], two, float], float, c_DistKernelMatrix[float, float]](deref(K.c_matrix), deref(c_NNdata), c_csplit, c_rsplit, deref(config.c_config),comm.ob_mpi)
        else:
            conf = Config(metric_type, K.dim(), m, k, s, stol, budget, sec_acc)
            conf.set_symmetry(sym)
            conf.set_adaptive_rank(adapt_ranks)
            with nogil:
                self.c_tree = c_compress[centersplit[c_DistKernelMatrix[float, float], two, float], randomsplit[c_DistKernelMatrix[float, float], two, float], float, c_DistKernelMatrix[float, float]](deref(K.c_matrix), deref(c_NNdata), c_csplit, c_rsplit, deref(conf.c_config),comm.ob_mpi)

    def evaluate_rids(self, DistData_RIDS rids):
        if not self.cStatus:
            raise Exception("You must run compress before running evaluate")
        result = DistData_RIDS(self.our_comm, m=rids.rows(), n=rids.cols(), mper=rids.local_rows())
        cdef RIDS_STAR_DistData[float]* bla
        with nogil:
            bla = Python_Evaluate[nnprune, km_float_tree, float](deref(self.c_tree), deref(rids.c_data))
        del result.c_data
        result.c_data = bla
        return result

    def evaluate(self, DistData_RIDS rids):
        return self.evaluate_rids(rids)

    def evaluate_rblk(self, DistData_RBLK rblk):
        if not self.cStatus:
            raise Exception("You must run compress before running evaluate")
        result = DistData_RBLK(self.our_comm, m=rblk.rows(), n=rblk.cols())
        cdef RBLK_STAR_DistData[float]* bla;
        with nogil:
            bla = Evaluate_Python_RBLK[nnprune, km_float_tree, float](deref(self.c_tree), deref(rblk.c_data))
        free(result.c_data)
        result.c_data = bla
        return result

    def evaluate_test(self, LocalData Xte, DistData_RIDS rids):
        result = LocalData( m= Xte.cols(), n = rids.cols())
        cdef Data[float]* bla; 
        with nogil:
            bla = TestMultiply[km_float_tree,float]( deref(self.c_tree), deref(Xte.c_data), deref(rids.c_data))
        
        free(result.c_data)
        result.c_data = bla
        return result
    	
    def evaluate_distributed_test(self, DistData_CBLK Xte, DistData_RIDS rids):
        result = LocalData( m= Xte.cols(), n = rids.cols())
        cdef Data[float]* bla; 
        with nogil:
            bla = TestMultiply[km_float_tree,float]( deref(self.c_tree), deref(Xte.c_data), deref(rids.c_data))
        
        free(result.c_data)
        result.c_data = bla
        return result


    def factorize(self, float reg):
        if not self.cStatus:
            raise Exception("You must run compress before running factorize")
        with nogil:
            DistFactorize[float, km_float_tree](deref(self.c_tree), reg)
        self.fStatus = 1

    def solve(self, LocalData w):
        #overwrites w! (also returns w)
        #Also its with a local copy?? not DistData
        if not self.fStatus:
            raise Exception("You must factorize before running solve")
        with nogil:
            DistSolve[float, km_float_tree](deref(self.c_tree), deref(w.c_data))
        return w

    def test_error(self,size_t ntest = 100,size_t nrhs = 10):
        with nogil:
            SelfTesting[km_float_tree]( deref(self.c_tree), ntest, nrhs)

def All_Nearest_Neighbors(MPI.Comm comm,size_t n, size_t k, localpoints, str metric="GEOMETRY_DISTANCE", leafnode=128):
    cdef STAR_CBLK_DistData[pair[float, size_t]]* NNList
    cdef randomsplit[c_DistKernelMatrix[float, float], two, float] c_rsplit
    cdef libmpi.MPI_Comm c_comm
   
    #Case 1: localpoints is a distributed data object 
    if isinstance(localpoints, DistData_CBLK):
        kernel = Kernel("GAUSSIAN")
        d = localpoints.rows()
        K = Distributed_Kernel_Matrix(comm, kernel, localpoints)
        c_rsplit.Kptr = K.c_matrix
        conf = Config(problem_size = n, metric_type=metric, neighbor_size = k, leaf_node_size = leafnode)
        c_comm = comm.ob_mpi
        with nogil:
            NNList = FindNeighbors_Python(deref(K.c_matrix), c_rsplit, deref(conf.c_config), c_comm, 10)
        PyNNList = DistDataPair(comm, n, d);
        free(PyNNList.c_data)
        PyNNList.c_data = NNList;
        return PyNNList

    #Case 2: localpoints is a local numpy array
    elif isinstance(localpoints, np.ndarray):
        d = localpoints.shape[0]
        DD_points = DistData_CBLK(comm, d, n, darr=localpoints)
        kernel = Kernel("GAUSSIAN")
        K = Distributed_Kernel_Matrix(comm, kernel, DD_points)
        c_rsplit.Kptr = K.c_matrix
        c_comm = comm.ob_mpi;
        conf = Config(problem_size = n, metric_type=metric, neighbor_size=k, leaf_node_size = leafnode)
        with nogil: 
            NNList = FindNeighbors_Python(deref(K.c_matrix), c_rsplit, deref(conf.c_config), c_comm, 10)
        PyNNList = DistDataPair(comm, n, d)
        free(PyNNList.c_data)
        PyNNList.c_data = NNList;
        return PyNNList


#Alternative (compact notation) GOFMM Kernel Matrix
cdef class KernelMatrix:
    @cython.nonecheck(False)
    def __cinit__(self, MPI.Comm comm, DistData_CBLK sources, DistData_CBLK targets=None, str kstring="GAUSSIAN", Config config=None, float bandwidth=1):
        self.comm_mpi = comm
        self.kernel = Kernel(kstring)
        self.kernel.set_bandwidth(bandwidth)
        self.config = config
        self.K = Distributed_Kernel_Matrix(comm, self.kernel, sources, targets)
        self.tree = KMTree(comm)
        self.is_compressed = 0
        self.is_factorized = 0
        self.size = sources.cols()
        self.d = sources.rows()
        self.nlocal = sources.nlocal() #len(self.get_local_gids())

    def compress(self, float stol=0.001, float budget=0.01, size_t m = 128, size_t k=64, size_t s = 32, bool sec_acc=True, str metric_type="ANGLE_DISTANCE", bool sym=True, bool adapt_ranks=True):
        with cython.nonecheck(True):
            if (self.config != None):
                self.tree.compress(self.comm_mpi, self.K, config=self.config)
            else:
                print("Configuration object not set. Compressing with default parameters...\n")
                self.tree.compress(self.comm_mpi, self.K, stol, budget, m, k, s, sec_acc, metric_type, sym, adapt_ranks)
        self.is_compressed = 1

    def solve(self, LocalData w, float l):
        if not self.is_factorized:
            print("Inverse factorization was not formed before calling solve. Factorizing now...\n")
            self.tree.factorize(l)
            self.is_factorized = 1
        if self.is_factorized:
            self.tree.solve(w)

    def set_config(self, Config conf):
        self.config = conf
        self.is_compressed = 0;

    def set_comm(self, MPI.Comm comm):
        self.comm_mpi = comm
        self.is_compressed = 0;

    def get_value(self, size_t i, size_t j):
        return self.K.get_value(i, j)

    def evaluate(self, DistData_RIDS rids):
        if (self.is_compressed):
            return self.tree.evaluate_rids(rids)
        elif (self.config != None):
            print("KernelMatrix is not compressed. Compressing with provided configuration object...\n")
            self.compress()
            self.is_compressed = 1
            return self.tree.evaluate_rids(rids)
        else:
            raise Exception("KernelMatrix must be compressed before evaluate is run. Please either run compress() or set the configuration object")

    def evaluate_test(self, LocalData Xte, DistData_RIDS rids):
        return self.tree.evaluateTest( Xte, rids)
	
    def evaluate_distributed_test(self, LocalData Xte, DistData_RIDS rids):
        return self.tree.evaluate_distributed_test( Xte, rids)

    def get_comm(self):
        return self.comm_mpi
        
    def get_tree(self):
        return self.tree
  
    def get_local_gids(self):
        return self.tree.get_gids()

    def set_custom_functions(self, f, g):
        self.isCompressed = 0
        return self.kernel.set_custom_function(f, g)

    def set_bandwidth(self, float b):
        self.isCompressed = 0
        self.kernel.set_bandwidth(b)

    def get_bandwidth(self):
        return self.kernel.get_bandwidth()

    def set_scale(self, float scal):
        self.kernel.set_scale(scal)
   
    def get_scale(self, float scal):
        return self.kernel.get_scale()

    def get_size(self):
        return self.size

    def get_local_size(self):
        return self.nlocal

    def test_error(self,size_t ntest = 100,size_t nrhs = 10):
        if(self.is_compressed):
            self.tree.test_error(ntest,nrhs)
        else:
            raise Exception("KernelMatrix must be compressed before error can be tested. Run compress().")

    def __getitem__(self, pos):
        if isinstance(pos, tuple) and len(pos) == 2:
            i, j = pos
            return self.get_value(i, j)
        else:
            raise Exception('KernelMatrix must be indexed in 2 dimensions')


