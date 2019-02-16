from Data cimport Data
#cimport as numpy as np
from cython.operator cimport dereference as deref



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
    #cpdef submatrix(self,np.ndarray[np.int, ndim=1] I not None,
    #        np.ndarray[np.int,ndim=1] J not None):

    #    # get sizes, initialize new PyData
    #    size_t ni = I.size()
    #    size_t nj = J.size()
    #    PyData sub = PyData(ni,nj)

    #    # call c_data's sub func
    #    sub.c_data = self.c_data(I,J)

    #    # return sub
    #    return sub
