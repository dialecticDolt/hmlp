from Splitters cimport *

cdef class centerSplit:
	cdef centersplit c_csplit
	def __cinit__(self, SPDMATRIX* K):
		self.c_csplit =  centersplit(K);

cdef class randomSplit:
	cdef randomsplit c_rsplit
	def __cinit(self, SPDMATRIX*K):
		self.c_rsplit = centersplit(K);


