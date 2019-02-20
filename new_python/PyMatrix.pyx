from Matrix cimport * 

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

        return m
    

#cdef class PyKernelMatrix:
#    cdef KernelMatrix[float]* c_matrix
#    
#    # constructor 
#    def __cinit__(self, size_t m = 0, size_t n = 0, str source_file = None, str target_file = None):
#        # read in sources, targets if availaible
#    

