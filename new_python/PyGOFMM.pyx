from Matrix cimport SPDMatrix, centersplit, randomsplit, Tree, dTree_t, sTree_t
from Matrix cimport Compress as c_compress
from Data cimport Data
from Config cimport *
from libcpp.pair cimport pair
from cython.operator cimport dereference as deref
cimport numpy as np


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

    # constructor
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

#cdef class PyTree:
#    cdef Tree* c_tree
#    def __cinit__(self):
#        self.c_tree = new Tree()
#    cdef copy(self, Tree *treeptr):
#        del self.c_tree
#        self.c_tree = treeptr

cdef class sPyTree:
    cdef sTree_t* c_stree
    def __cinit__(self):
         self.c_stree = new sTree_t()
    cdef copy(self, sTree_t *treeptr):
         del self.c_stree
         self.c_stree = treeptr

#def compress(PySPDMatrix py_matrix, PyData py_data, PyConfig py_config):
#    cdef centersplit c_csplit
#    cdef randomsplit c_rsplit
#    c_csplit.Kptr = py_matrix.c_matrix
#    c_rsplit.Kptr = py_matrix.c_matrix
#    py_tree = PyTree()
#    py_tree.copy(c_compress(py_matrix.c_matrix, deref(py_data.c_data), c_csplit, c_rsplit, py_config.c_config))
#   return py_tree

def compress(PySPDMatrix py_matrix, float stol, float budget, int m, int k, int s):
     py_tree = sPyTree()
     py_tree.copy(c_compress(py_matrix.c_matrix, stol, budget, m, k, s))
     return py_tree

