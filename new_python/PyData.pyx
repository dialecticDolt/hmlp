from Data cimport Data

cdef class PyData:
    cdef Data[float] c_data
	
    def __cinit__(self,size_t m = 0,size_t n = 0):
        self.c_data = Data[float](m,n)
	
    cpdef read(self, size_t m, size_t n, str filename):
        self.c_data.read(m, n,filename.encode())
    
    cpdef write(self,str filename):
        self.c_data.write(filename.encode())

    cpdef row(self):
        return self.c_data.row()
    
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
