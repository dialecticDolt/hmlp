from Matrix cimport SPDMatrix

cdef class PySPDMatrix:
    cdef SPDMatrix[float] c_matrix

    # constructor
    def __cinit__(self,size_t m = 0,size_t n = 0):
        self.c_matrix = SPDMatrix[float](m,n)
    
    cpdef read(self, size_t m, size_t n, str filename):
        self.c_matrix.read(m, n,filename.encode())
    
    cpdef row(self):
        return self.c_matrix.row()
    
    cpdef col(self):
        return self.c_matrix.col()
    
    cpdef size(self):
        return self.c_matrix.row() * self.c_matrix.col()

    cpdef getvalue(self,size_t m, size_t n):
        return self.c_matrix(m,n)



