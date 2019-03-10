#Import our functions
from DistData cimport *
from Runtime cimport *

#Import MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi

#Import from cython cpp
from libcpp.pair cimport pair
from libcpp cimport vector

#Import from cython c
from libc.math cimport sqrt
from libc.string cimport strcmp
from libc.stdlib cimport malloc, free

#Import from cython
from cython.operator cimport dereference as deref

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
#           - Add support for reading from distributed file
#           - "Template" on float/double fused type, need to decide on how we'll structure this

cdef class PyDistData:
    cdef CIRC_CIRC_f_DistData* c_data

    def __cinit__(self, size_t m, size_t n, int owner, MPI.Comm comm):
        self.c_data = new CIRC_CIRC_f_DistData(m, n, owner, comm.ob_mpi)

    def __dealloc__(self):
        print("Cython: Running __dealloc___ for PyData Object")
        self.c_data.clear()
        free(self.c_data)

    def rand(self, float a, float b):
        self.c_data.rand(a, b)
    
