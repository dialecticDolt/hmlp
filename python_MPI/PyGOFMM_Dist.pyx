#Import our functions
from DistData cimport *
from Runtime cimport *
from DistMatrix cimport *
from CustomKernel cimport *
from Config cimport *
from DistTree cimport *
from DistTree cimport Compress as c_compress
#Import MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi

#Import from cython: cpp
from libcpp.pair cimport pair
from libcpp cimport vector

#Import from cython: c
from libc.math cimport sqrt
from libc.string cimport strcmp
from libc.stdlib cimport malloc, free

#Import from cython
from cython.operator cimport dereference as deref

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
    
    def __cinit__(self, str metric_type, int problem_size, int leaf_node_size, int neighbor_size, int maximum_rank, float tolerance, float budget, bool secure_accuracy):
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

    def __cinit__(self, size_t m = 0, size_t n = 0):
        self.c_data = new Data[float](m, n)
    
    def __dealloc__( self ):
        print("Cython: Running __dealloc___ for PyData Object")
       # self.c_data.clear()
       # free(self.c_data)

    cpdef read(self, size_t m, size_t n, str filename):
        self.c_data.read(m, n,filename.encode())

    cpdef write(self,str filename):
        self.c_data.write(filename.encode())

    cpdef row(self):
        return self.c_data[0].row()

    cpdef col(self):
        return self.c_data.col()

    cpdef size(self):
        return self.c_data.size()

    cpdef getvalue(self,size_t m, size_t n):
        return self.c_data.getvalue(m,n)

    cpdef setvalue(self,size_t m, size_t n, float v):
        self.c_data.setvalue(m,n,v)

    cpdef rand(self,float a, float b ):
        self.c_data.rand(a, b)

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
#           - "Template" on Distribution Type. (have a pointer to BaseDistData that we case?)
#           - Add support for additional constructors
#                   - Look into how local vector distributes. Make wrapper to take "global" numpy array and spread it around. 
#                   - Add support for local numpy array
#           - Add support for reading from distributed file
#           - "Template" on float/double fused type, need to decide on how we'll structure this

cdef class PyDistData_CBLK:
    cdef STAR_CBLK_DistData[float]* c_data
    cdef MPI.Comm our_comm

    def __cinit__(self, MPI.Comm comm, size_t m, size_t n, str fileName=None, localdata=None):
        self.our_comm = comm
        cdef string fName
        if not (fileName or localdata): 
            self.c_data = new STAR_CBLK_DistData[float](m, n, comm.ob_mpi)
        elif fileName and not (localdata):
            fName = <string>fileName
            #TODO: Error handling 
            self.c_data = new STAR_CBLK_DistData[float](m, n, comm.ob_mpi, fName)
        elif localdata and not (fileName):
            if type(localdata) is PyData:
                self.c_data = new STAR_CBLK_DistData[float](m, n, deref(<Data[float]*>(PyData(localdata).c_data)), comm.ob_mpi)
            if isinstance(localdata, (np.ndarray, np.generic)):
                print("Loading local numpy arrays is not yet supported")
        else:
            print("Invalid constructor parameters")

    def __dealloc__(self):
        print("Cython: Running __dealloc___ for PyDistData Object")
        self.c_data.clear()
        free(self.c_data)
    
    #def loadCIDS(self, PyDistData_CIDS b):
    #    free(self.c_data)
    #    cdef STAR_CBLK_DistData[float] a = STAR_CIDS_DistData[float](b.rows(), b.cols(), self.our_comm.ob_mpi)
    #    a = (deref(b.c_data))
    #    self.c_data = &a

    #Fill the data object with random uniform data from the interval [a, b]
    def rand(self, float a, float b):
        self.c_data.rand(a, b)

    #TODO: Resolve ambigious overloading with Data[T]
    #def loadFile(self, size_t m, size_t n, str fileName):
    #    cdef string fName = <string>fileName
    #    self.c_data.read(m, n, fName)

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

    #THIS IS A LOCAL WRITE
    #def write(self, str fileName):
    #    cdef string fName = <string>fileName
    #    self.c_data.write(fName)


cdef class PyDistData_RBLK:
    cdef RBLK_STAR_DistData[float]* c_data
    cdef MPI.Comm our_comm
 
    def __cinit__(self, MPI.Comm comm, size_t m, size_t n, str fileName=None, localdata=None):
        #cdef string fName
        #if not (fileName or localdata): 
        self.c_data = new RBLK_STAR_DistData[float](m, n, comm.ob_mpi)
        #elif fileName and not (localdata):
        #    fName = <string>fileName
        #    #TODO: Error handling 
        #    self.c_data = new RBLK_STAR_DistData[float](m, n, comm.ob_mpi, fName)
        #elif localdata and not (fileName):
        #    if type(localdata) is PyData:
        #        self.c_data = new RBLK_STAR_DistData[float](m, n, deref(<Data[float]*>(PyData(localdata).c_data)), comm.ob_mpi)
        #    if isinstance(localdata, (np.ndarray, np.generic)):
        #        print("Loading local numpy arrays is not yet supported")
        #else:
        #    print("Invalid constructor parameters")

    def __dealloc__(self):
        print("Cython: Running __dealloc___ for PyDistData Object")
        self.c_data.clear()
        free(self.c_data)


    def loadRIDS(self, PyDistData_RIDS b):
        free(self.c_data)
        cdef RBLK_STAR_DistData[float]* a = new RBLK_STAR_DistData[float](b.rows(), b.cols(), self.our_comm.ob_mpi)
        a[0] = (deref(b.c_data))
        self.c_data = a

    #Fill the data object with random uniform data from the interval [a, b]
    def rand(self, float a, float b):
        self.c_data.rand(a, b)

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

#TODO: RIDS & CIDS DistData need a numpy constructor
cdef class PyDistData_RIDS:
    cdef RIDS_STAR_DistData[float]* c_data

#    def __cinit__(self, MPI.Comm comm, size_t m, size_t n, str fileName=None, localdata=None):
#        #cdef string fName
#        #if not (fileName or localdata): 
#        self.c_data = new RIDS_STAR_DistData[float](m, n, comm.ob_mpi)
#        #elif fileName and not (localdata):
#        #    fName = <string>fileName
#        #    #TODO: Error handling 
#        #    self.c_data = new RBLK_STAR_DistData[float](m, n, comm.ob_mpi, fName)
#        #elif localdata and not (fileName):
#        #    if type(localdata) is PyData:
#        #        self.c_data = new RBLK_STAR_DistData[float](m, n, deref(<Data[float]*>(PyData(localdata).c_data)), comm.ob_mpi)
#        #    if isinstance(localdata, (np.ndarray, np.generic)):
#        #        print("Loading local numpy arrays is not yet supported")
#        #else:
#        #    print("Invalid constructor parameters")
#
#    def __dealloc__(self):
#        print("Cython: Running __dealloc___ for PyDistData Object")
#        self.c_data.clear()
#        free(self.c_data)
#
#
#    def loadRBLK(self, PyDistData_RBLK b):
#        free(self.c_data)
#        cdef RIDS_STAR_DistData[float] a = RBLK_STAR_DistData[float](b.rows(), b.cols(), self.our_comm.ob_mpi)
#        a = (deref(b.c_data))
#        self.c_data = &a
#
#    #Fill the data object with random uniform data from the interval [a, b]
#    def rand(self, float a, float b):
#        self.c_data.rand(a, b)
#
#    def getCommSize(self):
#        return self.c_data.GetSize()
#
#    def getRank(self):
#        return self.c_data.GetRank()
#
#    def rows(self):
#        return self.c_data.row()
#
#    def cols(self):
#        return self.c_data.col()
#
#    def rows_local(self):
#        return self.c_data.row_owned()
#
#    def cols_local(self):
#        return self.c_data.col_owned()




cdef class PyDistPairData:
    cdef STAR_CBLK_DistData[pair[float, size_t]]* c_data

    def __cinit__(self, MPI.Comm comm, size_t m, size_t n, str fileName=None, localdata=None):
        cdef string fName
        if not (fileName or localdata): 
            self.c_data = new STAR_CBLK_DistData[pair[float, size_t]](m, n, comm.ob_mpi)
        elif fileName and not (localdata):
            fName = <string>fileName
            #TODO: Error handling 
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

    #TODO: Resolve ambigious overloading with Data[T]
    #def loadFile(self, size_t m, size_t n, str fileName):
    #    cdef string fName = <string>fileName
    #    self.c_data.read(m, n, fName)

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

    #THIS IS A LOCAL WRITE
    #def write(self, str fileName):
    #    cdef string fName = <string>fileName
    #    self.c_data.write(fName)

   
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

    def __cinit(self, MPI.Comm comm, PyKernel kernel, PyDistData_CBLK sources, PyDistData_CBLK targets=None):
        cdef size_t m, d, n
        m = sources.col()
        d = sources.row()
        n = m

        if targets is not None:
            n = targets.col()
            self.c_matrix = new DistKernelMatrix[float, float](m, n, d, deref(kernel.c_kernel), deref(sources.c_data), deref(targets.c_data), comm.ob_mpi)
        else:
            self.c_matrix = new DistKernelMatrix[float, float](m, d, deref(kernel.c_kernel), deref(sources.c_data), comm.ob_mpi)


#Python class for Kernel Matrix Tree
#ctypedef DTree[Setup[DistKernelMatrix[float, float], centersplit[DistKernelMatrix[float, float], two , float], float], NodeData[float]] km_float_tree

cdef class PyTreeKM:
    cdef Tree[Setup[DistKernelMatrix[float, float], centersplit[DistKernelMatrix[float, float], two , float], float], NodeData[float]]* c_tree 

    def __cinit__(self, MPI.Comm comm):
        self.c_tree = new Tree[Setup[DistKernelMatrix[float, float], centersplit[DistKernelMatrix[float, float], two , float], float], NodeData[float]](comm.ob_mpi)

    def __dealloc__(self):
        print("Cython: Running __dealloc__ for PyTreeKM")
        free(self.c_tree)

#TODO: Theres a namespace bug here. Compress is returning hmlp::tree::Compress for some reason. I need hmlp::mpitree::Compress. 

#    def compress(self, MPI.Comm comm, PyDistKernelMatrix K, float stol=0.001, float budget=0.01, size_t m=128, size_t k=64, size_t s=32, bool sec_acc=True, str metric_type="ANGLE_DISTANCE", bool sym=True, bool adapt_ranks=True, PyConfig config=None):
#        cdef centersplit[DistKernelMatrix[float, float], two, float] c_csplit
#        cdef randomsplit[DistKernelMatrix[float, float], two, float] c_rsplit
#        cdef STAR_CBLK_DistData[pair[float, size_t]]* c_NNdata = new STAR_CBLK_DistData[pair[float, size_t]](0, 0, comm.ob_mpi)
#        c_csplit.Kptr = K.c_matrix
#        c_rsplit.Kptr = K.c_matrix
#        if(config):
#            self.c_tree = c_compress[centersplit[DistKernelMatrix[float, float], two, float], randomsplit[DistKernelMatrix[float, float], two, float], float, DistKernelMatrix[float, float]](deref(K.c_matrix), deref(c_NNdata), c_csplit, c_rsplit, deref(config.c_config))
#        else:
#            conf = PyConfig(metric_type, K.row(), m, k, s, stol, budget, sec_acc)
#            conf.setSymmetry(sym)
#            conf.setAdaptiveRank(adapt_ranks)
#            self.c_tree = c_compress[centersplit[DistKernelMatrix[float, float], two, float], randomsplit[DistKernelMatrix[float, float], two, float], float, DistKernelMatrix[float, float]](deref(K.c_matrix), deref(c_NNdata), c_csplit, c_rsplit, deref(conf.c_config))



