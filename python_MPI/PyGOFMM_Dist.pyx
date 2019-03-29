#Import our functions
from DistData cimport *
from Runtime cimport *
from DistMatrix cimport *
from CustomKernel cimport *
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

#Import from cython: c
from libc.math cimport sqrt
from libc.string cimport strcmp
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf


#Import from cython
import cython
from cython.operator cimport dereference as deref
from cython.view cimport array as cvarray

#Numpy
import numpy as np
cimport numpy as np
np.import_array()

cdef class PyRuntime:
    cpdef int isInit

    def __cinit__(self):
        self.isInit = int(0)

    def __dealloc__(self):
        hmlp_finalize()

    cpdef init(self):
        cdef int arg_c=7
        cdef char **arg_v = <char **>malloc(7* sizeof(char*))
        hmlp_init(&arg_c, &arg_v)

    cpdef init_with_MPI(self, MPI.Comm comm):
        hmlp_init(comm.ob_mpi)

    cpdef set_num_workers(self, int nworkers):
        if self.isInit is 1:
            hmlp_set_num_workers(nworkers)

    cpdef run(self):
        hmlp_run()

    cpdef finalize(self):
        hmlp_finalize()

def convertMetric(str metric_type):
   if(metric_type == "GEOMETRY_DISTANCE"):
      m = int(0)
   elif(metric_type == "KERNEL_DISTANCE"):
      m = int(1)
   elif(metric_type == "ANGLE_DISTANCE"):
      m = int(2)
   elif(metric_type == "USER_DISTANCE"):
      m = int(3)
   return m

cdef class PyConfig:
    cdef Configuration[float]* c_config
    cpdef str metric_t
    
    def __cinit__(self, str metric_type="GEOMETRY_DISTANCE", int problem_size=2000, int leaf_node_size=128, int neighbor_size=64, int maximum_rank=128, float tolerance=0.0001, float budget=0.01, bool secure_accuracy=True):
        self.metric_t = metric_type
        m = convertMetric(metric_type) 
        m = int(m)
        self.c_config = new Configuration[float](m, problem_size, leaf_node_size, neighbor_size, maximum_rank, tolerance, budget, secure_accuracy)

    def setAll(self, str metric_type, int problem_size, int leaf_node_size, int neighbor_size, int maximum_rank, float tolerance, float budget, bool secure_accuracy):
        self.metric_t = metric_type
        m = convertMetric(metric_type)
        m = int(m)
        self.c_config.Set(m, problem_size, leaf_node_size, neighbor_size, maximum_rank, tolerance, budget, secure_accuracy)

    #TODO: Add getters and setters for all
    def setMetricType(self, metric_type):
        self.metric_t = metric_type
        m = convertMetric(metric_type)
        m = int(m)
        self.c_config.Set(m, self.getProblemSize(), self.getLeafNodeSizei(), self.getNeighborSize(), self.getMaximumRank(), self.getTolerance(), self.getBudget(), self.isSecure())

    def setNeighborSize(self, int nsize):
        self.c_config.Set(self.c_config.MetricType(), self.getProblemSize(), self.getLeafNodeSize(), nsize, self.getMaximumRank(), self.getTolerance(), self.getBudget(), self.isSecure())

    def setProblemSize(self, int psize):
        self.c_config.Set(self.c_config.MetricType(), psize, self.getLeafNodeSizei(), self.getNeighborSize(), self.getMaximumRank(), self.getTolerance(), self.getBudget(), self.isSecure())

    def setMaximumRank(self, int mrank):
        self.c_config.Set(self.c_config.MetricType(), self.getProblemSize(), self.getLeafNodeSize(), self.getNeighborSize(), mrank, self.getTolerance(), self.getBudget(), self.isSecure())

    def setTolerance(self, float tol):
        self.c_config.Set(self.c_config.MetricType(), self.getProblemSize(), self.getLeafNodeSize(), self.getNeighborSize(), self.getMaximumRank(), tol, self.getBudget(), self.isSecure())

    def setBudget(self, float budget):
        self.c_config.Set(self.c_config.MetricType(), self.getProblemSize(), self.getLeafNodeSize(), self.getNeighborSize(), self.getMaximumRank(), self.getTolerance(), budget, self.isSecure())

    def setSecureAccuracy(self, bool status):
        self.c_config.Set(self.c_config.MetricType(), self.getProblemSize(), self.getLeafNodeSize(), self.getNeighborSize(), self.getMaximumRank(), self.getTolerance(), self.getBudget(), status)
        
    def getMetricType(self):
        return self.metric_t

    def getMaximumRank(self):
        return self.c_config.MaximumRank()

    def getNeighborSize(self):
        return self.c_config.NeighborSize()

    def getProblemSize(self):
        return self.c_config.ProblemSize()

    def getMaximumDepth(self):
        return self.c_config.getMaximumDepth()

    def getLeafNodeSize(self):
        return self.c_config.getLeafNodeSize()

    def getTolerance(self):
        return self.c_config.Tolerance()

    def getBudget(self):
        return self.c_config.Budget()

    def isSymmetric(self):
        return self.c_config.IsSymmetric()

    def isAdaptive(self):
        return self.c_config.UseAdaptiveRanks()

    def isSecure(self):
        return self.c_config.SecureAccuracy()

    def setLeafNodeSize(self, int leaf_node_size):
        self.c_config.setLeafNodeSize(leaf_node_size)

    def setAdaptiveRank(self, bool status):
        self.c_config.setAdaptiveRanks(status)

    def setSymmetry(self, bool status):
        self.c_config.setSymmetric(status)


#Wrapper for locally stored PyData class. 
#   TODO:
#           - Update this to functionality of Shared Memory PyGOFMM (I just copied over a subset of functions to test things)
#           - Update this to new "templated" version
#           - I'm not sure what copy functions are working/to keep??

cdef class PyData:
    cdef Data[float]* c_data

    #TODO: Add error handling if the user gives the wrong array sizes (at the moment I'm overriding)
    @cython.boundscheck(False)
    def __cinit__(self, size_t m=0, size_t n=0, str fileName=None, float[::1, :] darr=None, float[:] arr=None, PyData data=None):
        cdef string fName
        cdef vector[float] vec
        cdef int vec_sz
        if fileName and not (data or darr!=None or arr!=None):
            #Load data object from file
            fName = <string>fileName.encode('utf-8')
            with nogil:
                self.c_data = new Data[float](m, n)
                self.c_data.read(m, n, fName)
        elif data and not (fileName or darr!=None or arr!=None):
            #Deep copy from existing data object
            with nogil:
                self.c_data = new Data[float](deref(data.c_data))
        elif darr!=None and not (fileName or data!=None or arr!=None):
            #Load data object from 2d numpy array
            m = <size_t>darr.shape[0]
            n = <size_t>darr.shape[1]
            vec_sz = <int> (m * n)
            with nogil:
                vec.assign(&darr[0, 0], &darr[0,0] + vec_sz)
                self.c_data = new Data[float](m, n, &darr[0, 0], True)
        elif arr!=None and not (fileName or data!=None or darr!=None):
            #Load data object from numpy array
            m = <size_t>arr.size
            n = <size_t>1
            vec_sz = len(arr)
            with nogil:
                vec.assign(&arr[0], &arr[0] + vec_sz)
                self.c_data = new Data[float](m, n, vec)
        else:
            #Create empty Data object
            with nogil:
                self.c_data = new Data[float](m, n)

    def __dealloc__( self ):
        print("Cython: Running __dealloc___ for PyData Object")
        # self.c_data.clear()
        free(self.c_data)

    #TODO: Add error handling in case non integers are passed
    def __getitem__(self, pos):
        if isinstance(pos, int):
            return self.c_data.getvalue(<size_t>pos,<size_t>0)
        #elif isinstance(pos, int) and self.c_data.row() == 1:
        #     return self.c_data.getvalue(<size_t>0, <size_t>pos)
        elif isinstance(pos, tuple) and len(pos) == 2:
            i, j = pos
            return self.c_data.getvalue(i, j)
        else:
            raise Exception('PyData can only be indexed in 1 or 2 dimensions')

    #TODO: Add error handling in case non integers are passed
    def __setitem__(self, pos, float v):
        if isinstance(pos, int):
            self.c_data.setvalue(<size_t>pos,<size_t>0, v)
        #elif isinstance(pos, int) and self.c_data.row() == 1:
        #    self.c_data.setvalue(<size_t>0, <size_t>pos, v)
        elif not isinstance(pos, int) and len(pos) == 2:
            i, j = pos
            self.c_data.setvalue(<size_t>i, <size_t>j, v)
        else:
            raise Exception('PyData can only be indexed in 1 or 2 dimensions')

    @cython.boundscheck(False)
    @classmethod 
    def FromNumpy(cls,np.ndarray[float, ndim=2, mode="c"] arr_np):
        # get sizes
        cdef size_t m,n
        m = <size_t> arr_np.shape[0]
        n = <size_t> arr_np.shape[1]
        
        # construct std::vector
        #cdef vector[float] arr_cpp = vector[float](m*n)
        #arr_cpp.assign(&arr_np[0,0], &arr_np[-1,-1])

        # construct PyData obj
        cpdef PyData ret = cls(m,n)
        #cdef Data[float]* bla = new Data[float](m,n,arr_cpp)
        cdef Data[float]* bla 
        with nogil:
            bla = new Data[float](m,n,&arr_np[0,0],True)
        ret.c_data = bla
        return ret

    def toNumpy(self):
        cdef float* data_ptr = self.c_data.rowdata(0)
        cdef float[:] mv = <float[:self.size()]> data_ptr
        np_arr = np.asarray(mv)
        np_arr.resize(self.row(),self.col())
        return np_arr

    @cython.boundscheck(False)
    def submatrix(self,np.ndarray[np.intp_t, ndim=1] I not None,
        np.ndarray[np.intp_t,ndim=1] J not None):

        # define memory views?
        cdef np.intp_t [:] Iview = I
        cdef np.intp_t [:] Jview = J


        cdef size_t ci,cj

        # get sizes, initialize new PyData
        cdef size_t ni = <size_t> I.size
        cdef size_t nj = <size_t> J.size
        cdef Data[float]* subdata = new Data[float](ni,nj)
        cdef float tmp

        # begin loops
        for ci in range(ni):
            for cj in range(nj):
                tmp = self.c_data.getvalue( <size_t> Iview[ci], <size_t> Jview[cj] )
                subdata.setvalue(<size_t> ci,<size_t> cj,tmp)

        # new Pydata object
        cpdef PyData sub = PyData(ni,nj)

        # call c_data's sub func
        sub.c_data = subdata

        # return sub
        return sub

    cpdef read(self, size_t m, size_t n, str filename):
        cdef string fName
        fName = <string>filename.encode('utf-8')
        with nogil:
            self.c_data.read(m, n,fName)

    cpdef write(self,str filename):
        self.c_data.write(filename.encode())

    cpdef rows(self):
        return self.c_data[0].row()

    cpdef cols(self):
        return self.c_data.col()

    cpdef size(self):
        return self.c_data.size()

    cpdef setvalue(self,size_t m, size_t n, float v):
        self.c_data.setvalue(m,n,v)
    
    cpdef getvalue(self, size_t m, size_t n):
        return self.c_data.getvalue(m, n)

    cpdef rand(self,float a, float b ):
        with nogil:
            self.c_data.rand(a, b)

    cpdef randn(self, float mu, float std):
        with nogil:
            self.c_data.randn(mu, std)

    cpdef randspd(self, float a, float b):
        with nogil:
            self.c_data.randspd(a, b)

    cpdef display(self):
        self.c_data.Print()

    cpdef HasIllegalValue(self):
        return self.c_data.HasIllegalValue()

    cpdef MakeCopy(self):
        # get my data stuff
        cdef Data[float]* cpy = new Data[float](deref(self.c_data) )
        # put into python obj
        cpdef PyData bla = PyData(self.row(), self.col())
        bla.c_data = cpy
        return bla

    cdef deepcopy(self,PyData other):
        del self.c_data
        self.c_data = new Data[float]( deref(other.c_data) )



#Python Class for Distributed Data Object - Columnwise
#   TODO:
#           - "Template" on float/double fused type, need to decide on how we'll structure this

cdef class PyDistData_CBLK:
    cdef STAR_CBLK_DistData[float]* c_data
    cdef MPI.Comm our_comm

    #TODO: Fix loading from local PyData objects
    @cython.boundscheck(False)
    def __cinit__(self, MPI.Comm comm, size_t m=0, size_t n=0, str fileName=None, float[::1, :] darr=None, float[:] arr=None, PyData data=None):
        cdef string fName
        cdef vector[float] vec
        cdef int vec_sz
        if fileName and not (data or darr!=None or arr!=None):
            #Load data object from file
            fName = <string>fileName.encode('utf-8')
            with nogil:
                self.c_data = new STAR_CBLK_DistData[float](m, n, comm.ob_mpi, fName)
        elif data and not (fileName or darr!=None or arr!=None):
            #From local copy of PyData object TODO TODO TODO Same error as dist data deep copy constructors (arguments don't match)
            raise Exception("This is currently broken")
            self.c_data = new STAR_CBLK_DistData[float](m, n, deref(<Data[float]*>(PyData(data).c_data)), comm.ob_mpi)
        elif darr!=None and not (fileName or data!=None or arr!=None):
            #Load data object from 2d numpy array
            vec_sz = darr.shape[0] * darr.shape[1]
            with nogil:
                vec.assign(&darr[0, 0], &darr[0,0] + vec_sz)
                self.c_data = new STAR_CBLK_DistData[float](m, n, vec, comm.ob_mpi)
        elif arr!=None and not (fileName or data!=None or darr!=None):
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
            raise Exception('PyData can only be indexed in 1 or 2 dimensions')

    def __dealloc__(self):
        print("Cython: Running __dealloc___ for PyDistData Object")
        free(self.c_data)
    
    def rand(self, float a=0.0, float b=1.0):
        with nogil:
            self.c_data.rand(a, b)

    def display(self):
        self.c_data.Print()

    def randn(self, float m=0.0, float s=1.0):
        with nogil:
            self.c_data.randn(m, s)

    def getCommSize(self):
        return self.c_data.GetSize()

    def getRank(self):
        return self.c_data.GetRank()

    def rows(self):
        return self.c_data.row()

    def cols(self):
        return self.c_data.col()

    def rows_local(self):
        return self.c_data.row_owned()

    def cols_local(self):
        return self.c_data.col_owned()

cdef class PyDistData_RBLK:
    cdef RBLK_STAR_DistData[float]* c_data
    cdef MPI.Comm our_comm

    #TODO: Fix loading from local PyData objects
    @cython.boundscheck(False)
    def __cinit__(self, MPI.Comm comm, size_t m=0, size_t n=0, str fileName=None, float[::1, :] darr=None, float[:] arr=None, PyData data=None):
        cdef string fName
        cdef vector[float] vec
        cdef int vec_sz

        if fileName and not (data or darr!=None or arr!=None):
            #Load data object from file
            raise Exception("RBLK does not support loading from a file. Please load a RIDS and convert if you need a RBLK")
        elif data and not (fileName or darr!=None or arr!=None):
            #From local copy of PyData object TODO: Fix this, same problem as CBLK
            self.c_data = new RBLK_STAR_DistData[float](m, n, deref(<Data[float]*>(PyData(data).c_data)), comm.ob_mpi)
        elif darr!=None and not (fileName or data!=None or arr!=None):
            raise Exception('RBLK does not currently support loading from 2D numpy arrays')
            #Load data object from 2D numpy array
            vec_sz = darr.shape[0] + darr.shape[1]
            with nogil:
                vec.assign(&darr[0, 0], &darr[0,0] + vec_sz)
                self.c_data = new RBLK_STAR_DistData[float](m, n, vec, comm.ob_mpi)
        elif arr!=None and not (fileName or data!=None or darr!=None):
            #Load data object from numpy array
            vec.assign(&arr[0], &arr[0] + len(arr))
            with nogil:
                self.c_data = new RBLK_STAR_DistData[float](m, n, vec, comm.ob_mpi)
        else:
            #Create empty Data object
            with nogil:
                self.c_data = new RBLK_STAR_DistData[float](m, n, comm.ob_mpi)

    def __dealloc__(self):
        print("Cython: Running __dealloc___ for PyDistData Object")
        free(self.c_data)

    #TODO: Add a class method for this
    def loadRIDS(self, PyDistData_RIDS b):
        with nogil:
            self.c_data[0] = (deref(b.c_data))

    def __getitem__(self, pos):
        if isinstance(pos, int) and self.c_data.col() == 1:
            return deref(self.c_data)(<size_t>pos,<size_t>0)
        elif len(pos) == 2:
            i, j = pos
            return deref(self.c_data)(<size_t>i, <size_t>j)
        else:
            raise Exception('PyData can only be indexed in 1 or 2 dimensions')

    def rand(self, float a=0.0, float b=1.0):
        with nogil:
            self.c_data.rand(a, b)

    def display(self):
        self.c_data.Print()

    def randn(self, float m=0.0, float s=1.0):
        with nogil:
            self.c_data.randn(m, s)

    def getCommSize(self):
        return self.c_data.GetSize()

    def getRank(self):
        return self.c_data.GetRank()

    def rows(self):
        return self.c_data.row()

    def cols(self):
        return self.c_data.col()

    def rows_local(self):
        return self.c_data.row_owned()

    def cols_local(self):
        return self.c_data.col_owned()


cdef class PyDistData_RIDS:
    cdef RIDS_STAR_DistData[float]* c_data
    cdef MPI.Comm our_comm
    cdef map[size_t, size_t] rid2row

    @cython.boundscheck(False)
    def __cinit__(self, MPI.Comm comm,size_t m=0, size_t n=0, size_t mper = 0, str fileName=None, int[:] iset=None, float[::1,:] darr=None, float[:] arr=None, PyTreeKM tree=None, PyData data=None, PyDistData_RIDS ddata=None):
        cdef string fName
        rank = comm.Get_rank()
        if (arr!=None):
            mper = len(arr)
        if (darr!=None):
            mper = darr.shape[0]
        cdef int[:] a = np.arange(rank*mper, rank*mper+mper).astype('int32')
        cdef vector[size_t] vec
        cdef vector[float] dat
        cdef int vec_sz

        # assign owned indices (vec)
        if tree:
            vec = tree.c_tree.getGIDS()
        elif iset==None and tree==None:
            vec.assign(&a[0], &a[0] + len(a))
        else:
            vec.assign(&iset[0], &iset[0] + len(iset))
            
        # call DistData constructor 
        if fileName and not (data or darr!=None or arr!=None):
            #Load data object from file
            fName = <string>fileName.encode('utf-8')
            with nogil:
                self.c_data = new RIDS_STAR_DistData[float](m, n, fName, comm.ob_mpi)
        elif data and not (fileName or darr!=None):
            #From local copy of PyData object
            raise Exception("RIDS does not yet support loading from localdata...it will soon")
        elif arr!=None and not (fileName or darr!=None):
            #Load data object from 1D numpy array
            dat.assign(&arr[0], &arr[0] + len(arr))
            self.c_data = new RIDS_STAR_DistData[float](m,n,vec,dat,comm.ob_mpi)

        elif darr!=None and not (fileName or data!=None):
            #Load data object from 2D numpy array
            vec_sz = darr.shape[0] * darr.shape[1] 
            with nogil:
                dat.assign(&darr[0,0], &darr[0,0] + vec_sz)
                self.c_data = new RIDS_STAR_DistData[float](m,n,vec,dat,comm.ob_mpi)

        elif ddata and not (fileName or darr!=None or data):
            with nogil:
                self.c_data = new RIDS_STAR_DistData[float](deref(ddata.c_data), comm.ob_mpi)
        else:
            with nogil:
                self.c_data = new RIDS_STAR_DistData[float](m, n, vec, comm.ob_mpi)

        self.rid2row = self.c_data.getMap()

    def __dealloc__(self):
        print("Cython: Running __dealloc___ for PyDistData Object")
        free(self.c_data)

    def rand(self, float a=0.0, float b=1.0):
        self.c_data.rand(a, b)

    def display(self):
        self.c_data.Print()

    def randn(self, float m=0.0, float s=1.0):
        self.c_data.randn(m, s)
   
    def loadRBLK(self, PyDistData_RBLK b):
        with nogil:
            self.c_data[0] = (deref(b.c_data))

    def redistribute(self, PyDistData_RIDS b):
        with nogil:
            self.c_data[0] = (deref(b.c_data))

    def toNumpy(self):
        cdef float* data_ptr = self.c_data.rowdata(0)
        cdef float[:] mv = <float[:self.size()]> data_ptr
        np_arr = np.asarray(mv)
        np_arr.resize(self.rows_local(),self.cols_local())
        return np_arr

    def toArray(self):
        cdef float* local_data
        local_data = self.c_data.rowdata(0)
        cdef float[:] mv = <float[:self.c_data.size()]> local_data
        np_arr = np.asarray(mv, order='F', dtype='float32').reshape((self.rows_local(),self.cols_local()),order='F')
        return np_arr

    def __setitem__(self, pos, float v):
        i, j = pos
        i = self.rid2row[i]
        self.c_data.setvalue(<size_t>i, <size_t>j, v)


#@staticmethod
    #def Loop2d(MPI.Comm comm, float[:,:] darr):
    #    cdef int rank = comm.Get_rank()
    #    cdef float bla
    #    cdef int m,n,i,itot
    #    m = darr.shape[0]
    #    n = darr.shape[1]
    #    itot = m * n
    #    comm.barrier()
    #    
    #    if (rank == 0):
    #        printf("2d looper %d\n",itot)
    #        #for i in range(m):
    #        #    for j in range(n):
    #        #        bla = deref(&darr[i,j])
    #        #        printf("%f\n", bla)
    #        #        #print(darr[i,j])

    #        for i in range(itot):
    #            bla = deref(&darr[0,0] + i)
    #            printf("%d %f\n",i, bla)
    #        printf("\n")
    #            
    #    comm.barrier()


    #@staticmethod
    #def Loop1d(MPI.Comm comm, float[:] arr):
    #    cdef int rank = comm.Get_rank()
    #    cdef float bla
    #    
    #    comm.barrier()
    #            
    #    if (rank == 0):
    #        printf("1d looper %d\n",len(arr))
    #        for i in range(len(arr)):
    #            bla = deref(&arr[i])
    #            printf("%d %f\n", i,bla)
    #            #print(<float> deref( <float *>( &arr[0] + i) ))
    #            #print(arr[i])
    #        printf("\n")

    #    comm.barrier()


    def getRank(self):
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
            raise Exception('PyData can only be indexed in 1 or 2 dimensions')

    def getRIDS(self):
        cdef vector[size_t] rids_vec
        rids_vec = self.c_data.getRIDS()
        return np.asarray(rids_vec)

    def rows_local(self):
        return self.c_data.row_owned()

    def cols_local(self):
        return self.c_data.col_owned()

cdef class PyDistPairData:
    cdef STAR_CBLK_DistData[pair[float, size_t]]* c_data

    #TODO: Loading from file, how to handle tuples?
    def __cinit__(self, MPI.Comm comm, size_t m, size_t n, str fileName=None, localdata=None):
        cdef string fName
        if not (fileName or localdata): 
            with nogil:
                self.c_data = new STAR_CBLK_DistData[pair[float, size_t]](m, n, comm.ob_mpi)
        elif fileName and not (localdata):
            fName = <string>fileName.encode('utf-8')
            with nogil:
                self.c_data = new STAR_CBLK_DistData[pair[float, size_t]](m, n, comm.ob_mpi, fName)
        elif localdata and not (fileName):
            if type(localdata) is PyData:
                self.c_data = new STAR_CBLK_DistData[pair[float, size_t]](m, n, deref(<Data[pair[float, size_t]]*>(PyData(localdata).c_data)), comm.ob_mpi)
            if isinstance(localdata, (np.ndarray, np.generic)):
                print("Loading local numpy arrays is not yet supported")
        else:
            print("Invalid constructor parameters")

    def __dealloc__(self):
        print("Cython: Running __dealloc___ for PyDistData Object")
        self.c_data.clear()
        free(self.c_data)
    
    def __getitem__(self, pos):
        if isinstance(pos, int) and self.c_data.col() == 1:
            return deref(self.c_data)(<size_t>pos,<size_t>1)
        elif len(pos) == 2:
            i, j = pos
            return deref(self.c_data)(<size_t>i, <size_t>j)
        else:
            raise Exception('PyData can only be indexed in 1 or 2 dimensions')

    def toNumpy(self):
        cdef int local_cols
        local_cols = self.cols_local()
        cdef int local_rows
        local_rows = self.rows_local()
        cdef float[:, :] mv_distances = np.empty((local_rows, local_cols), dtype='float32')
        cdef size_t[:, :] mv_gids = np.empty((local_rows, local_cols), dtype='uintp')
        for i in range(local_rows):
            for j in range(local_cols):
                mv_distances[i, j] = self[i, j][0]
                mv_gids[i, j] = self[i, j][1]
        np_distances = np.asarray(mv_distances)
        np_gids = np.asarray(mv_gids)
        return (np_distances, np_gids)

    #TODO: Pass back as a numpy array of dtype=object (tuple) (Minimize copying)
    #def toNumpy2(self):
    #    cdef int local_cols
    #    local_cols = self.cols_local()
    #    cdef int local_rows
    #    local_rows = self.rows_local()
    #    cdef pair[float, size_t]* data_ptr = self.c_data.rowdata(0)
    #    cdef pair[float, size_t][:] mv = <pair[float, size_t][:(local_cols*local_rows)] data_ptr
    #    np_arr = np.asarray(mv)
    #    np_arr.resize(local_rows, local_cols)
    #    return np_arr
        

    def getCommSize(self):
        return self.c_data.GetSize()

    def getRank(self):
        return self.c_data.GetRank()

    def rows(self):
        return self.c_data.row()

    def cols(self):
        return self.c_data.col()

    def rows_local(self):
        return self.c_data.row_owned()

    def cols_local(self):
        return self.c_data.col_owned()

#Python Class for Kernel Evaluation Object kernel_s
#   TODO:
#           -"Template" on float/double fused type
cdef class PyKernel:
    cdef kernel_s[float,float]* c_kernel

    # constructor 
    def __cinit__(self,str kstring="GAUSSIAN"):
       self.c_kernel = new kernel_s[float,float]()
       k_enum = PyKernel.getKernelType(kstring)
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
    def getKernelType(str kstring):
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
        elif(kstring == "USER_DEFINE"):
            m = int(9)
        else:
            raise ValueError("Kernel type not found.")

        return m

    # Gaussian param set/get
    def setBandwidth(self,float _scal):
        self.c_kernel.scal = -0.5 / (_scal *_scal)

    def getBandwidth(self):
        f = -0.5 / self.c_kernel.scal
        return sqrt(f)

    def setScal(self,float _scal):
        self.c_kernel.scal = _scal

    def getScal(self):
        return self.c_kernel.scal

    def setCustomFunction(self, f, g):
        self.user_element_function = f
        self.user_matrix_function = g


cdef class PyDistKernelMatrix:
    cdef DistKernelMatrix[float, float]* c_matrix

    def __cinit__(self, MPI.Comm comm, PyKernel kernel, PyDistData_CBLK sources, PyDistData_CBLK targets=None):
        cdef size_t m, d, n
        m = sources.c_data.col()
        d = sources.c_data.row()
        n = m

        if targets is not None:
            n = targets.col()
            with nogil:
                self.c_matrix = new DistKernelMatrix[float, float](m, n, d, deref(kernel.c_kernel), deref(sources.c_data), deref(targets.c_data), comm.ob_mpi)
        else:
            with nogil:
                self.c_matrix = new DistKernelMatrix[float, float](m, d, deref(kernel.c_kernel), deref(sources.c_data), comm.ob_mpi)

    def dim(self):
        return self.c_matrix.dim()

    def __getitem__(self, pos):
        if len(pos) == 2:
            i, j = pos
            return deref(self.c_matrix)(<size_t>i, <size_t>j)

    def getvalue(self, i, j):
        return self[i, j]

#Python class for Kernel Matrix Tree
ctypedef Tree[Setup[DistKernelMatrix[float, float], centersplit[DistKernelMatrix[float, float], two , float], float], NodeData[float]] km_float_tree

cdef class PyTreeKM:
    cdef km_float_tree* c_tree 
    cdef MPI.Comm our_comm
    cdef int cStatus
    cdef int fStatus

    def __cinit__(self, MPI.Comm comm):
        with nogil:
            self.c_tree = new km_float_tree(comm.ob_mpi)
        self.our_comm = comm
        self.cStatus = 0
        self.fStatus = 0

    def __dealloc__(self):
        print("Cython: Running __dealloc__ for PyTreeKM")
        free(self.c_tree)

    def getGIDS(self):
        cdef vector[size_t] gidvec
        with nogil:
            gidvec = self.c_tree.getGIDS()
        np_arr = np.asarray(gidvec, dtype='int32')
        return np_arr

    def compress(self, MPI.Comm comm, PyDistKernelMatrix K, float stol=0.001, float budget=0.01, size_t m=128, size_t k=64, size_t s=32, bool sec_acc=True, str metric_type="ANGLE_DISTANCE", bool sym=True, bool adapt_ranks=True, PyConfig config=None):
        cdef centersplit[DistKernelMatrix[float, float], two, float] c_csplit
        cdef randomsplit[DistKernelMatrix[float, float], two, float] c_rsplit
        cdef STAR_CBLK_DistData[pair[float, size_t]]* c_NNdata = new STAR_CBLK_DistData[pair[float, size_t]](0, 0, comm.ob_mpi)
        c_csplit.Kptr = K.c_matrix
        c_rsplit.Kptr = K.c_matrix
        self.cStatus=1
        if(config):
            with nogil:
                self.c_tree = c_compress[centersplit[DistKernelMatrix[float, float], two, float], randomsplit[DistKernelMatrix[float, float], two, float], float, DistKernelMatrix[float, float]](deref(K.c_matrix), deref(c_NNdata), c_csplit, c_rsplit, deref(config.c_config),comm.ob_mpi)
        else:
            conf = PyConfig(metric_type, K.dim(), m, k, s, stol, budget, sec_acc)
            conf.setSymmetry(sym)
            conf.setAdaptiveRank(adapt_ranks)
            with nogil:
                self.c_tree = c_compress[centersplit[DistKernelMatrix[float, float], two, float], randomsplit[DistKernelMatrix[float, float], two, float], float, DistKernelMatrix[float, float]](deref(K.c_matrix), deref(c_NNdata), c_csplit, c_rsplit, deref(conf.c_config),comm.ob_mpi)

    def evaluateRIDS(self, PyDistData_RIDS rids):
        if not self.cStatus:
            raise Exception("You must run compress before running evaluate")
        result = PyDistData_RIDS(self.our_comm, m=rids.rows(), n=rids.cols(), mper=rids.rows_local())
        cdef RIDS_STAR_DistData[float]* bla
        with nogil:
            bla = Evaluate_Python_RIDS[nnprune, km_float_tree, float](deref(self.c_tree), deref(rids.c_data))
        free(result.c_data)
        result.c_data = bla
        return result


    def evaluateRBLK(self, PyDistData_RBLK rblk):
        if not self.cStatus:
            raise Exception("You must run compress before running evaluate")
        result = PyDistData_RBLK(self.our_comm, m=rblk.rows(), n=rblk.cols())
        cdef RBLK_STAR_DistData[float]* bla;
        with nogil:
            bla = Evaluate_Python_RBLK[nnprune, km_float_tree, float](deref(self.c_tree), deref(rblk.c_data))
        free(result.c_data)
        result.c_data = bla
        return result

    def factorize(self, float reg):
        if not self.cStatus:
            raise Exception("You must run compress before running factorize")
        with nogil:
            DistFactorize[float, km_float_tree](deref(self.c_tree), reg)
        self.fStatus = 1

    def solve(self, PyData w):
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

def FindAllNeighbors(MPI.Comm comm,size_t n, size_t k, localpoints, str metric="GEOMETRY_DISTANCE", leafnode=128):
    cdef STAR_CBLK_DistData[pair[float, size_t]]* NNList
    cdef randomsplit[DistKernelMatrix[float, float], two, float] c_rsplit
    cdef libmpi.MPI_Comm c_comm
    if isinstance(localpoints, PyDistData_CBLK):
        kernel = PyKernel("GAUSSIAN")
        d = localpoints.rows()
        print(d)
        K = PyDistKernelMatrix(comm, kernel, localpoints)
        c_rsplit.Kptr = K.c_matrix
        conf = PyConfig(problem_size = n, metric_type=metric, neighbor_size = k, leaf_node_size = leafnode)
        c_comm = comm.ob_mpi
        with nogil:
            NNList = FindNeighbors_Python(deref(K.c_matrix), c_rsplit, deref(conf.c_config), c_comm, 10)
        PyNNList = PyDistPairData(comm, n, d);
        free(PyNNList.c_data)
        PyNNList.c_data = NNList;
        return PyNNList
    elif isinstance(localpoints, np.ndarray):
        d = localpoints.shape[0]
        DD_points = PyDistData_CBLK(comm, d, n, darr=localpoints)
        kernel = PyKernel("GAUSSIAN")
        K = PyDistKernelMatrix(comm, kernel, DD_points)
        c_rsplit.Kptr = K.c_matrix
        c_comm = comm.ob_mpi;
        conf = PyConfig(problem_size = n, metric_type=metric, neighbor_size=k, leaf_node_size = leafnode)
        with nogil: 
            NNList = FindNeighbors_Python(deref(K.c_matrix), c_rsplit, deref(conf.c_config), c_comm, 10)
        PyNNList = PyDistPairData(comm, n, d)
        free(PyNNList.c_data)
        PyNNList.c_data = NNList;
        return PyNNList

#Alternative (compact notation) GOFMM Kernel Matrix
cdef class KernelMatrix:
    cdef PyKernel kernel
    cdef PyTreeKM tree
    cdef PyDistKernelMatrix K
    cdef PyConfig config
    cdef MPI.Comm comm_mpi
    cdef int is_compressed 
    cdef int is_secacc
    cdef int is_factorized    

    @cython.nonecheck(False)
    def __cinit__(self, MPI.Comm comm, PyDistData_CBLK sources, PyDistData_CBLK targets=None, str kstring="GAUSSIAN", PyConfig conf=None, float bandwidth=1):
        self.comm_mpi = comm
        self.kernel = PyKernel(kstring)
        self.kernel.setBandwidth(bandwidth)
        self.config = conf
        self.K = PyDistKernelMatrix(comm, self.kernel, sources, targets)
        self.tree = PyTreeKM(comm)
        self.is_compressed = 0
        self.is_factorized = 0

    def compress(self, float stol=0.001, float budget=0.01, size_t m = 128, size_t k=64, size_t s = 32, bool sec_acc=True, str metric_type="ANGLE_DISTANCE", bool sym=True, bool adapt_ranks=True):
        with cython.nonecheck(True):
            if (self.config != None):
                self.tree.compress(self.comm_mpi, self.K, config=self.config)
            else:
                self.tree.compress(self.comm_mpi, self.K, stol, budget, m, k, s, sec_acc, metric_type, sym, adapt_ranks)
        self.is_compressed = 1

    def solve(self, PyData w, float l):
        if not self.is_factorized:
            self.tree.factorize(l)
            self.is_factorized = 1
        if self.is_factorized:
            self.tree.solve(w)

    def setConfig(self, PyConfig conf):
        self.config = conf
        if(self.is_compressed):
            print("Recompressing K with new configuration parameters...")
            self.compress()
    
    def setComm(self, MPI.Comm comm):
        self.comm_mpi = comm

    def getValue(self, size_t i, size_t j):
        return self.K.getvalue(i, j)

    def evaluate(self, PyDistData_RIDS rids):
        if (self.is_compressed):
            return self.tree.evaluateRIDS(rids)
        elif (self.config!=None):
            self.compress()
            self.is_compressed = 1
            return self.tree.evaluateRIDS(rids)
        else:
            raise Exception("KernelMatrix must be compressed before evaluate is run. Please either run compress() or provide a configuration object")

    def getComm(self):
        return self.comm_mpi
        
    def getTree(self):
        return self.tree
  
    def setCustomFunctions(self, f, g):
        return self.kernel.setCustomFunction(f, g)

    def setBandwidth(self, float b):
        return  self.kernel.setBandwidth(b)
