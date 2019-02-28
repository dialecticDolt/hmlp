from Runtime cimport *
from Matrix cimport *
from Matrix cimport Compress as c_compress
from Data cimport Data
from Config cimport *
from libcpp.pair cimport pair
from libc.math cimport sqrt
from cython.operator cimport dereference as deref
cimport numpy as np

cdef extern from "Python.h":
    char* PyUnicode_AsUTF8(object unicode)

from libc.stdlib cimport malloc, free
from libc.string cimport strcmp

cdef char ** to_cstring_array(list_str):
    cdef char **ret = <char **>malloc(len(list_str) * sizeof(char *))
    for i in xrange(len(list_str)):
        ret[i] = PyUnicode_AsUTF8(list_str[i])
    return ret



cdef class PyRuntime:
    cpdef int isInit

    def __cinit__( self ):
        self.isInit = int(0)

    def __dealloc__( self ):
        hmlp_finalize()

    cpdef init( self ):
        # create dummy args -- doesn't work properly with none
        cdef int arg_c = 7
        cdef char **arg_v = <char **>malloc(7 * sizeof(char *))
        #cdef char** arg_v = to_cstring_array(str_list)
        hmlp_init(&arg_c, &arg_v)
        #self.isInit = int(1)

    cpdef set_num_workers( self, int nworkers ):
        if self.isInit is 1:
            hmlp_set_num_workers( nworkers )

    cpdef run( self ):
        hmlp_run()

    cpdef finalize( self ):
        hmlp_finalize()
                          


cdef class PyConfig:
    cdef Configuration[float]* c_config
    cpdef str metric_t

    def __cinit__(self, str metric_type, int problem_size, int leaf_node_size, int neighbor_size, int maximum_rank, float tolerance, float budget, bool secure_accuracy):
        self.metric_t = metric_type
        if(metric_type == "GEOMETRY_DISTANCE"):
            m = int(0)
        elif(metric_type == "KERNEL_DISTANCE"):
            m = int(1)
        elif(metric_type == "ANGLE_DISTANCE"):
            m = int(2)
        elif(metric_type == "USER_DISTANCE"):
            m = int(3)
        self.c_config = new Configuration[float](m, problem_size, leaf_node_size, neighbor_size, maximum_rank, tolerance, budget, secure_accuracy)

    def getMetricType(self):
        return self.metric_t

    def getProblemSize(self):
        return self.c_config.ProblemSize()

    def getMaximumDepth(self):
        return self.c_config.getMaximumDepth()

    def getLeafNodeSize(self):
        return self.c_config.getLeafNodeSize()

    def isSymmetric(self):
        return self.c_config.IsSymmetric()

    def isAdaptive(self):
        return self.c_config.UseAdaptiveRanks()

    def isSecure(self):
        return self.c_config.SecureAccuracy()

    def setLeafNodeSize(self, int leaf_node_size):
        self.c_config.setLeafNodeSize(leaf_node_size)


cdef class PyData:
    cdef Data[float]* c_data

    def __cinit__(self,size_t m = 0,size_t n = 0):
        self.c_data = new Data[float](m,n)

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
        self.c_data.rand(a,b)
    
    cdef copy(self, Data[float] ptr):
        del self.c_data
        self.c_data = &ptr

    # TODO pass in numpy and make a data object?? try with [:]
    # not sure it is necessary, so going to leave this for later

    # TODO access submatrix through inputting numpy vectors
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


cdef class PySPDMatrix:
    cdef SPDMatrix[float]* c_matrix
    #TODO: Add option to make double precision matrix. 
    #cdef SPDMatrix[double]* dc_matrix
    
    def __cinit__(self,size_t m = 0,size_t n = 0):
       self.c_matrix = new SPDMatrix[float](m,n)
    
    cpdef read(self, size_t m, size_t n, str filename):
       self.c_matrix.read(m, n,filename.encode())
    
    cpdef row(self):
        return self.c_matrix.row()
    
    cpdef col(self):
        return self.c_matrix.col()
    
    cpdef size(self):
        return self.c_matrix.row() * self.c_matrix.col()

    cpdef getvalue(self,size_t m, size_t n):
        return self.c_matrix[0](m,n)

    def randspd(self, float a, float b):
        self.c_matrix.randspd(a, b)

cdef class PyKernel:
    cdef kernel_s[float,float]* c_kernel

    # constructor 
    def __cinit__(self,str kstring="GAUSSIAN"):
       self.c_kernel = new kernel_s[float,float]()
       k_enum = PyKernel.GetKernelTypeEnum(kstring)
       self.c_kernel.SetKernelType(k_enum)

    # static method for handling enum
    @staticmethod
    def GetKernelTypeEnum(str kstring):
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
    def SetBandwidth(self,float _scal):
        self.c_kernel.scal = -0.5 / (_scal *_scal)
    
    def GetBandwidth(self):
        f = -0.5 / self.c_kernel.scal
        return sqrt(f)

    def SetScal(self,float _scal):
        self.c_kernel.scal = _scal
    
    def GetScal(self):
        return self.c_kernel.scal

cdef class PyKernelMatrix:
    cdef KernelMatrix[float]* c_matrix

    def __cinit__(self,PyData sources,PyKernel kernel,PyData targets = None):
        # get m, d
        cdef size_t m,d,n
        m = sources.col() # sources should be input d x N
        d = sources.row()
        n = m
        
        # handle non-symmetric case, call constructor
        if targets is not None:
            n = targets.col()
            self.c_matrix = new KernelMatrix[float](m,n,d,deref(kernel.c_kernel),deref(sources.c_data),deref(targets.c_data))
        else:
            self.c_matrix = new KernelMatrix[float](m,n,d,deref(kernel.c_kernel),deref(sources.c_data))

    cpdef dim(self):
        return self.c_matrix.dim()
    
    cpdef size(self):
        return self.c_matrix.dim() * self.c_matrix.dim()

    cpdef getvalue(self,size_t m, size_t n):
        return self.c_matrix[0](m,n)

# Type defs to make life easier
#ctypedef Tree[Setup[SPDMatrix[float],centersplit[SPDMatrix[float],two,float],float],NodeData[float]] spd_float_tree

# GOFMM tree
cdef class PyTreeSPD:
    cdef Tree[Setup[SPDMatrix[float],centersplit[SPDMatrix[float],two,float],float],NodeData[float]]* c_tree
    #cdef spd_float_tree* c_tree

    # Initializer test
    def __cinit__(self):
        self.c_tree = new Tree[Setup[SPDMatrix[float],centersplit[SPDMatrix[float],two,float],float],NodeData[float]]()
        #self.c_tree = new spd_float_tree()

    # GOFMM compress
    def PyCompress(self,PySPDMatrix K, float stol, float budget, size_t m, size_t k, size_t s,bool sec_acc=True):
        # call real life compress
        self.c_tree = Compress[float, SPDMatrix[float]](deref(K.c_matrix),
                stol, budget, m, k, s,sec_acc)

    def PyEvaluate(self, PyData w):
        result = PyData()
        result.copy(Evaluate[use_runtime, use_opm_task, nnprune, cache, Tree[Setup[SPDMatrix[float], centersplit[SPDMatrix[float], two, float], float], NodeData[float]], float](deref(self.c_tree), deref(w.c_data)))

        #result.copy(Evaluate[use_runtime, use_opm_task,nnprune,cache,spd_float_tree,float](deref(self.c_tree), deref(w.c_data)))

        return result

    def PyFactorize(self,float reg):
        Factorize[float,  Tree[Setup[SPDMatrix[float],centersplit[SPDMatrix[float],two,float],float],NodeData[float]]]( deref(self.c_tree), reg)

        #Factorize[float,spd_float_tree]( deref(self.c_tree), reg)






#This would need to be templated on setup and nodetype
#cdef class PyTree:
#    cdef Tree* c_tree
#    def __cinit__(self):
#        self.c_tree = new Tree()
#    cdef copy(self, Tree *treeptr):
#        del self.c_tree
#        self.c_tree = treeptr

#These are single and double precision wrappers for Tree
cdef class sPyTree:
    cdef sTree_t* c_stree
    def __cinit__(self):
         self.c_stree = new sTree_t()
    cdef copy(self, sTree_t *treeptr):
         del self.c_stree
         self.c_stree = treeptr

cdef class dPyTree:
    cdef dTree_t* c_dtree
    def __cinit__(self):
         self.c_dtree = new dTree_t()
    cdef copy(self, dTree_t *treeptr):
         del self.c_dtree
         self.c_dtree = treeptr


#PyMatrix will serve the same functionality (&more) 
#Keep single and double precision SPDMatrix wrappers just in case
cdef class sPySPDMatrix:
    cdef sSPDMatrix_t *c_matrix
    
    def __cinit(self):
       self.c_matrix = new sSPDMatrix_t()

    def row(self):
        return self.c_matrix.row()

    def col(self):
        return self.c_matrix.col()

    def size(self):
        return self.c_matrix.row() * self.c_matrix.col()

    def getvalue(self,size_t i):
        return self.getvalue(i)

cdef class dPySPDMatrix:
    cdef dSPDMatrix_t *c_matrix

    def __cinit(self):
       self.c_matrix = new dSPDMatrix_t()

    def row(self):
        return self.c_matrix.row()

    def col(self):
        return self.c_matrix.col()

    def size(self):
        return self.c_matrix.row() * self.c_matrix.col()

    def getvalue(self,size_t i):
        return self.getvalue(i)

#def compress(PySPDMatrix py_matrix, float stol, float budget):
#    py_tree = sPyTree()
#    py_tree.copy(c_compress(deref(py_matrix.c_matrix), stol, budget))
#    return py_tree


