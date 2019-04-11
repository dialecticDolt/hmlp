from Runtime cimport *
from Config cimport *

cdef class PyRuntime:
    cdef int isInit

cdef class PyConfig:
    cdef Configutration[float]* c_config
    cpdef str metric_t


