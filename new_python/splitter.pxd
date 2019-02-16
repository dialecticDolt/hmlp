from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

#This wont work without SPDMATRIX being imported

cdef extern "${CMAKE_SOURCE_DIR}/gofmm/gofmm.hpp" namespace "gofmm":
	struct centersplit:
		cdef SPDMATRIX *Kptr
		centersplit()
		~centersplit()
		centersplit(SPDMATRIX*)
	
	struct randomsplit:
		cdef SPDMATRIX *Kptr
		randomsplit()
		~randomsplit()
		randomsplit(SPDMATRIX*)


