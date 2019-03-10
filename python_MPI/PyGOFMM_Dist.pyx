#Import our functions
from DistData cimport *
from Runtime cimport *
from DistMatrix cimport *
from CustomKernel cimport *

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
        self.c_data.clear()
        free(self.c_data)

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



#Python Class for Distributed Data Object
#   TODO:
#           - "Template" on Distribution Type. (have a pointer to BaseDistData that we case?)
#           - Add support for additional constructors
#                   - Look into how local vector distributes. Make wrapper to take "global" numpy array and spread it around. 
#                   - Add support for local numpy array
#           - Add support for reading from distributed file
#           - "Template" on float/double fused type, need to decide on how we'll structure this

cdef class PyDistData:
    cdef STAR_CBLK_f_DistData* c_data

    def __cinit__(self, MPI.Comm comm, size_t m, size_t n, int owner=(-1), str fileName=None, localdata=None):
        cdef string fName = <string>fileName
        if owner == (-1) and not (fileName or localdata): 
            self.c_data = new STAR_CBLK_f_DistData(m, n, owner, comm.ob_mpi)
        if fileName and not (owner!=(-1) or localdata):
            self.c_data = new STAR_CBLK_f_DistData(m, n, comm.ob_mpi, fName)
        if localdata and not (owner!=(-1) or fileName):
            if type(localdata) is PyData:
                self.c_data = new STAR_CBLK_f_DistData(m, n, deref(<Data[float]*>(PyData(localdata).c_data)), comm.ob_mpi)
            if isinstance(localdata, (np.ndarray, np.generic)):
                print("Loading local numpy arrays is not yet supported")
        else:
            print("Invalid constructor parameters")

    def __dealloc__(self):
        print("Cython: Running __dealloc___ for PyData Object")
        self.c_data.clear()
        free(self.c_data)

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

    
#Python Class for Kernel Evaluation Object kernel_s
#   TODO:
#           -"Template" on float/double fused type
cdef class PyKernel:
    cdef kernel_s[float,float]* c_kernel

    # constructor 
    def __cinit__(self,str kstring="GAUSSIAN"):
       self.c_kernel = new kernel_s[float,float]()
       k_enum = PyKernel.GetKernelTypeEnum(kstring)
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
    cdef DistKernelMatrix* c_matrix

    def __cinit(self, MPI.Comm comm, PyKernel kernel, PyDistData sources, PyDistData targets=None):
        cdef size_t m, d, n
        m = sources.col()
        d = sources.row()
        n = m

        if targets is not None:
            n = targets.col()
            self.c_matrix = new DistKernelMatrix(m, n, d, deref(kernel.c_kernel), deref(sources.c_data), deref(targets.c_data), comm.ob_mpi)
        else:
            self.c_matrix = new DistKernelMatrix(m, d, deref(kernel.c_kernel), deref(sources.c_data), comm.ob_mpi) 
