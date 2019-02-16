from Data cimport Data
cimport numpy as np
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
