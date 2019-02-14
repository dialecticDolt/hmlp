from Data cimport Data

cdef class PyData:
	cdef Data[float] c_data
	
	def __cinit__(self):
		self.c_data = Data[float]()
	
	def __cinit__(self, int m, int n):
		self.c_data = Data[float](m, n)

