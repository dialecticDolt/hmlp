from Data cimport Data

cdef class PyData:
    cdef Data[float] c_data
	
    def __cinit__(self,size_t m = 0,size_t n = 0):
        self.c_data = Data[float](m,n)
	
	# TODO -- can we specify type info for filename?
    cpdef read(self, size_t m, size_t n, filename):
        self.c_data.read(m, n,filename)
    
	# TODO -- can we specify type info for filename?
    cpdef write(self, size_t m, size_t n, filename):
        self.c_data.write(filename)

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

